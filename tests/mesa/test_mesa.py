from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from mesa.budget import BudgetManager
from mesa.canonical_grid import CanonicalGrid, map_view
from mesa.config import CHANNELS, MESAConfig
from mesa.engine import MESAInferenceEngine
from mesa.evidence_aggregator import EvidenceAggregator, text_evidence
from mesa.views import VisualViewGenerator
from reacquisition.region_selector import GridGeometry
from xiaomi_runtime.adapter import MODEL_SHA256, Prediction, text_hash
from xiaomi_runtime.otsl import to_html
from xiaomi_runtime.processor import ProcessorLimits

RAW = '<fcel>A<fcel>B<nl><fcel>C<fcel>D<nl>'


class MockRuntime:
    def __init__(self, failures=0):
        self.load_count = 0
        self.calls = []
        self.failures = failures
        self.model_hash = MODEL_SHA256

    def infer(self, table_id, image, *, method, view_type, max_visual_tokens, deadline):
        self.load_count = 1
        self.calls.append(view_type)
        if self.failures:
            self.failures -= 1
            raise RuntimeError('synthetic transient failure')
        w, h = ProcessorLimits().native_size(image.size)
        tokens = w * h // 1024
        assert tokens <= max_visual_tokens
        return Prediction(RAW, to_html(RAW), table_id, method, view_type,
                          input_size=image.size, processed_size=(w, h), image_grid_thw=[[1, h // 16, w // 16]],
                          visual_tokens=tokens, output_hash=text_hash(RAW))


@pytest.mark.parametrize('channels', [('global',), ('highres',), ('horizontal',), ('vertical',), ('tiles',), CHANNELS])
def test_channel_flow_single_model_and_cache(channels, tmp_path):
    runtime = MockRuntime()
    config = MESAConfig(channels=channels)
    engine = MESAInferenceEngine(runtime, config, tmp_path)
    image = Image.new('RGB', (320, 240), 'white')
    result = engine.run('dev:test', image)
    assert result['raw'] == RAW and runtime.load_count == 1 and runtime.model_hash == MODEL_SHA256
    assert {c.split('_')[0] for c in runtime.calls} == set(config.enabled)
    calls = len(runtime.calls)
    assert engine.run('dev:test', image) == result
    assert len(runtime.calls) == calls
    # Recover from saved per-view predictions, without recomputing any successful view.
    (tmp_path / text_hash('dev:test') / 'result.json').unlink()
    recovered = engine.run('dev:test', image)
    assert recovered['raw'] == RAW and len(runtime.calls) == calls


def test_views_roundtrip_overlap_and_native_metadata():
    image = Image.new('RGB', (701, 499), 'white')
    views = VisualViewGenerator(MESAConfig()).generate(image, 'source')
    assert views[0].prepare(image) is image
    for view in views:
        x0, y0, x1, y1 = view.crop_box
        assert 0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height
        assert view.prepare(image).size == view.plan.actual_size
        for processed in (False, True):
            for point in ((x0, y0), (x1, y1), ((x0+x1)/2, (y0+y1)/2)):
                assert view.to_original(view.from_original(point, processed=processed), processed=processed) == pytest.approx(point)
        scale = view.plan.applied_scale
        assert abs(view.plan.actual_size[0] - view.plan.original_size[0]*scale) <= .5
        assert abs(view.plan.actual_size[1] - view.plan.original_size[1]*scale) <= .5
    strips = [v for v in views if v.channel_id == 'horizontal']
    assert strips[0].crop_box[2] > strips[1].crop_box[0]
    assert all(v.pieces[0].source_bbox[1] == 0 for v in views if v.channel_id == 'vertical')
    assert any(v.crop_box[1] > 0 for v in views if v.channel_id == 'vertical')
    tiles = [v for v in views if v.channel_id == 'tiles']
    assert tiles[0].crop_box[2] > tiles[1].crop_box[0]


@pytest.mark.parametrize('grid,count', [('2x2',4), ('3x3',9)])
def test_tiles(grid,count):
    views = VisualViewGenerator(MESAConfig(channels=('tiles',),tile_grid=grid)).generate(Image.new('RGB',(600,600)), 'a')
    assert sum(v.channel_id == 'tiles' for v in views) == count


@pytest.mark.parametrize('field,value', [('overlap',0), ('overlap',.9), ('max_latency_per_table',float('nan')),
    ('detail_scale',2), ('max_tiles_per_table',10), ('skip_channels',('global',)), ('max_visual_tokens_per_view',32)])
def test_config_rejections(field,value):
    with pytest.raises(ValueError):
        MESAConfig(**{field:value})


def test_all_yaml_configs():
    paths = list(Path('configs/mesa').glob('*.yaml'))
    assert len(paths) == 7
    for path in paths:
        MESAConfig.load(path)


def test_retry_budget_and_skip(tmp_path):
    runtime = MockRuntime(failures=1)
    engine = MESAInferenceEngine(runtime, MESAConfig(channels=('global',)), tmp_path)
    result = engine.run('dev:test', Image.new('RGB',(256,256)))
    assert result['budget']['views'] == 2
    assert result['budget']['visual_tokens'] == 128
    runtime = MockRuntime()
    result = MESAInferenceEngine(runtime, MESAConfig(max_total_visual_tokens=64), tmp_path/'bounded').run(
        'dev:test',Image.new('RGB',(256,256)))
    assert runtime.calls == ['global_0']
    assert any(e['status']=='skipped' for e in result['events'])
    assert result['raw'] == RAW


def test_budget_all_limits():
    cfg=MESAConfig()
    view=VisualViewGenerator(cfg).generate(Image.new('RGB',(256,256)), 'a')[-1]
    assert BudgetManager(views=cfg.max_views_per_table).rejection(view,cfg) == 'view_budget_exhausted'
    assert BudgetManager(tiles=cfg.max_tiles_per_table).rejection(view,cfg) == 'tile_budget_exhausted'
    assert BudgetManager(elapsed=cfg.max_latency_per_table).rejection(view,cfg) == 'latency_budget_exhausted'
    assert BudgetManager(visual_tokens=cfg.max_total_visual_tokens).rejection(view,cfg) == 'total_visual_token_budget_exhausted'


def test_fusion_exception_and_identity_change(tmp_path):
    class Broken:
        def aggregate(self,*args):
            raise ValueError('deliberate test')
    engine=MESAInferenceEngine(MockRuntime(),MESAConfig(fusion=True),tmp_path)
    engine.aggregator=Broken()
    image=Image.new('RGB',(256,256))
    result=engine.run('dev:fail',image)
    assert result['fusion']['reason']=='fusion_failed_retain_global' and result['raw']==RAW
    with pytest.raises(ValueError,match='identity'):
        engine.run('dev:fail',Image.new('RGB',(256,256),'red'))


def test_canonical_spans_blank_and_empty_rows():
    raw='<fcel>A<lcel><fcel>B<nl><ucel><xcel><ecel><nl>'
    grid=CanonicalGrid.from_otsl(raw,'tiles','t')
    assert grid.native.status=='RESOLVED'
    first=grid.cells[0]
    assert (first.row_end,first.col_end,first.rowspan,first.colspan)==(2,2,2,2)
    assert first.source_channel=='tiles'
    assert CanonicalGrid.from_otsl('<ecel><ecel><nl><ecel><ecel><nl>','global','g').native.status=='RESOLVED'
    assert CanonicalGrid.from_otsl('<ucel><nl>','global','g').native.status=='UNRESOLVED'


def test_spatial_mapping_different_dimensions_and_split_span():
    image=Image.new('RGB',(200,200))
    view=VisualViewGenerator(MESAConfig(channels=('horizontal',),padding=0)).generate(image,'i')[1]
    view=replace(view,crop_box=(0,0,100,200),mapping_quality='HIGH')
    global_grid=CanonicalGrid.from_otsl(RAW,'global','g')
    local=CanonicalGrid.from_otsl('<fcel>A<nl><fcel>C<nl>','horizontal','h')
    geometry=GridGeometry((0.,100.,200.),(0.,100.,200.),True,'synthetic')
    mapped=map_view(global_grid,local,view,geometry)
    assert mapped['status']=='MAPPED' and mapped['cells'][1]['global']==(1,0)
    assert map_view(global_grid,local,replace(view,crop_box=(0,0,80,200)),geometry)['status']=='UNRESOLVED'
    span=CanonicalGrid.from_otsl('<fcel>A<lcel><nl><fcel>C<fcel>D<nl>','global','g')
    assert map_view(span,local,view,geometry)['reason']=='global_span_crosses_crop_boundary'


def test_text_evidence_never_prefers_length_or_changes_numbers():
    assert text_evidence('1,000','10000')['numeric_equal'] is False
    assert text_evidence('A B','AB')['whitespace_only'] is True
    assert text_evidence('A','A longer')['decision']=='retain_global_text'


def test_aggregation_can_accept_verified_local_structure():
    raw='<ecel><lcel><fcel>A<nl><fcel>B<fcel>C<fcel>D<nl>'
    candidate='<ecel><ecel><fcel>A<nl><fcel>B<fcel>C<fcel>D<nl>'
    image=Image.new('RGB',(301,201),'white')
    draw=ImageDraw.Draw(image)
    for x in (0,100,200,300):draw.line((x,0,x,200),fill='black')
    for y in (0,100,200):draw.line((0,y,300,y),fill='black')
    cfg=MESAConfig(channels=('highres',),fusion=True)
    view=VisualViewGenerator(cfg).generate(image,'a')[1]
    pred=Prediction(candidate,to_html(candidate),'a','MESA',view.view_id)
    result=EvidenceAggregator().aggregate(raw,[(view,pred)],image,cfg)
    assert result['changed'] and result['raw']==candidate
    # A text conflict vetoes repair, including a longer auxiliary answer.
    pred=replace(pred,raw=candidate.replace('A','A longer'))
    assert EvidenceAggregator().aggregate(raw,[(view,pred)],image,cfg)['raw']==raw


def test_skip_channels_and_global_failure(tmp_path):
    config=MESAConfig(skip_channels=('highres','tiles'))
    assert config.enabled==('global','horizontal','vertical')
    engine=MESAInferenceEngine(MockRuntime(failures=10),MESAConfig(),tmp_path)
    with pytest.raises(RuntimeError,match='Global Original unavailable'):
        engine.run('dev:missing',Image.new('RGB',(256,256)))
    assert (tmp_path/text_hash('dev:missing')/'failure.json').exists()


def test_late_auxiliary_is_discarded_and_global_retained(tmp_path):
    class Late(MockRuntime):
        def infer(self,*args,**kwargs):
            p=super().infer(*args,**kwargs)
            return replace(p,error='deadline_exceeded',truncated=True) if kwargs['view_type']!='global_0' else p
    result=MESAInferenceEngine(Late(),MESAConfig(channels=('highres',)),tmp_path).run(
        'dev:late',Image.new('RGB',(256,256)))
    assert result['raw']==RAW and len(result['cache_files'])==1
    assert sum(e['status']=='failed' for e in result['events'])==6


def test_cache_tampering_is_detected(tmp_path):
    engine=MESAInferenceEngine(MockRuntime(),MESAConfig(channels=('global',)),tmp_path)
    image=Image.new('RGB',(256,256))
    engine.run('dev:cache',image)
    cached=tmp_path/text_hash('dev:cache')/'views/global_0.json'
    cached.write_text(cached.read_text()+' ')
    with pytest.raises(ValueError,match='Cached view changed'):
        engine.run('dev:cache',image)


def test_unknown_inflight_cost_is_not_reset(tmp_path):
    import json
    runtime=MockRuntime()
    engine=MESAInferenceEngine(runtime,MESAConfig(channels=('highres',)),tmp_path)
    image=Image.new('RGB',(256,256))
    engine.run('dev:interrupted',image)
    folder=tmp_path/text_hash('dev:interrupted')
    (folder/'result.json').unlink()
    (folder/'views/highres_1.75.json').unlink()
    state=json.loads((folder/'budget.json').read_text())
    state['in_flight']=True
    (folder/'budget.json').write_text(json.dumps(state))
    calls=len(runtime.calls)
    result=engine.run('dev:interrupted',image)
    assert len(runtime.calls)==calls and result['raw']==RAW
    assert any(e.get('reason')=='latency_budget_exhausted' for e in result['events'])


def test_isolated_tile_cannot_replace_global():
    image=Image.new('RGB',(300,200),'white')
    cfg=MESAConfig(channels=('tiles',),fusion=True)
    view=VisualViewGenerator(cfg).generate(image,'a')[1]
    local='<fcel>longer unrelated text<nl>'
    pred=Prediction(local,to_html(local),'a','MESA',view.view_id)
    result=EvidenceAggregator().aggregate(RAW,[(view,pred)],image,cfg)
    assert result['raw']==RAW and not result['changed']
    assert result['evidence'][0]['spatial']['status']=='UNRESOLVED'

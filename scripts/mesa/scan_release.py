#!/usr/bin/env python3
"""v2 publication checks without changing the older publication implementation."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from orchestration.publication import content_issues

BASE='e743c68dff189258558e9872ebee3742c517bc09'
PREFIXES=('src/mesa/','tests/mesa/','configs/mesa/','scripts/mesa/','docs/mesa/','reports/mesa_v2/')


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True)


def issues(name,data):
    # Only the new report directory is added. All binary, size and secret checks
    # are the unchanged publication scanner's checks on the exact blob bytes.
    path=Path(name)
    alias='reports/mesa/'+path.name if path.parent==Path('reports/mesa_v2') and path.suffix=='.md' else name
    return content_issues(alias,data)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    failures={};seen=set()
    changed=set(git('diff','--name-only',BASE,'HEAD').splitlines())
    staged=git('diff','--cached','--name-only').splitlines();changed.update(staged)
    for name in changed:
        if not name.startswith(PREFIXES):failures[name]=['outside requested v2 modification scope']
    for name in staged:
        data=subprocess.check_output(['git','show',':'+name],cwd=ROOT)
        found=issues(name,data)
        if found:failures[name]=found
    for revision in git('rev-list','--all').splitlines():
        for line in git('ls-tree','-r',revision).splitlines():
            metadata,name=line.split('\t',1);blob=metadata.split()[2]
            if (name,blob) in seen:continue
            seen.add((name,blob))
            data=subprocess.check_output(['git','cat-file','blob',blob],cwd=ROOT)
            found=issues(name,data)
            if found:failures[f'{revision}:{name}']=found
    result={'passed':not failures,'changed':sorted(changed),'staged':staged,'history_blobs':len(seen),'failures':failures}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'history_blobs':len(seen),'failures':failures}))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()

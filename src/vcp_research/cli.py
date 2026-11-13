from __future__ import annotations
import argparse
import json
from pathlib import Path
from .core import run,Config


def main():
    p=argparse.ArgumentParser(description="VCP vs momentum causal OHLCV experiment")
    p.add_argument("--data",type=Path,default=Path("data"))
    p.add_argument("--out",type=Path,default=Path("results"))
    p.add_argument("--train-end",default="2019-12-31")
    p.add_argument("--validation-end",default="2022-12-31")
    p.add_argument("--test-start",default="2023-01-01")
    p.add_argument("--test-end",default="2026-11-01")
    p.add_argument("--min-dollar-volume",type=float,default=2_000_000)
    args=p.parse_args()
    cfg=Config(train_end=args.train_end,validation_end=args.validation_end,
               test_start=args.test_start,test_end=args.test_end,min_dollar_volume=args.min_dollar_volume)
    r=run(args.data,args.out,cfg)
    print(json.dumps(r,ensure_ascii=False,indent=2))

if __name__ == "__main__":main()
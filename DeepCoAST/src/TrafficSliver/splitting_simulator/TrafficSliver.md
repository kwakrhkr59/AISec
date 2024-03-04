<TrafficSliver 분할된 파일 경로(BigEnough/singlesite)>
/scratch/TrafficSliver
├ BigEnough
│  ├ path2
│  ├ path3
│  └ path5
│      ├ mon
│      └ unmon
│           └ ts
└ singlesite
    ├ path2
    ├ path3
    └ path5
        ├ mon_amazon
        ├ mon_amazon
        └ unmon
            └ ts

- scheme: batched_weighted_random
- circuits: 2 / 3 / 5
- circuits_min: 2 / 3 / 5
- weights: 0.5_0.5 / 0.33_0.33_0.34 / 0.2_0.2_0.2_0.2_0.2
- range: 50_70
- alpha: 1,1 / 1,1,1 / 1,1,1,1,1
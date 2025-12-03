# ================================================================================

# FEATURE IMPORTANCE ANALYSIS FOR WILDFIRE BURN SEVERITY

# ================================================================================

Dataset shape: (2210, 202)
Total samples: 2210
Total feature columns: 193

---

# ================================================================================

# MISSING DATA ANALYSIS

# ================================================================================

Features with missing data:

```text
                 column  missing_count  missing_pct
26                dndbi            520        23.53
13                 dbsi            520        23.53
69                mndwi            520        23.53
27                dndvi            520        23.53
25                 dnbr            520        23.53
24               dmndwi            520        23.53
36            greenBand            520        23.53
76                  nbr            520        23.53
77                 ndbi            520        23.53
78                 ndvi            520        23.53
79              nirBand            520        23.53
151           swir2Band            520        23.53
150           swir1Band            520        23.53
6                   bsi            520        23.53
124             redBand            520        23.53
5              blueBand            520        23.53
135               rlp_6            276        12.49
121              rdgh_4             96         4.34
114                po_2             96         4.34
128           relelev_4             96         4.34
127          relelev_32             96         4.34
115               po_32             96         4.34
116            profc_16             96         4.34
120             profc_8             96         4.34
126           relelev_2             96         4.34
125          relelev_16             96         4.34
123              rdgh_6             96         4.34
117             profc_2             96         4.34
118            profc_32             96         4.34
122              rdgh_5             96         4.34
119             profc_4             96         4.34
0              aspct_16             96         4.34
113             planc_8             96         4.34
112             planc_4             96         4.34
1               aspct_2             96         4.34
97   pisrdir_2021-01-22             96         4.34
98   pisrdir_2021-02-22             96         4.34
99   pisrdir_2021-03-22             96         4.34
100  pisrdir_2021-04-22             96         4.34
101  pisrdir_2021-05-22             96         4.34
102  pisrdir_2021-06-22             96         4.34
103  pisrdir_2021-07-22             96         4.34
104  pisrdir_2021-08-22             96         4.34
105  pisrdir_2021-09-22             96         4.34
106  pisrdir_2021-10-22             96         4.34
107  pisrdir_2021-11-22             96         4.34
108  pisrdir_2021-12-22             96         4.34
109            planc_16             96         4.34
110             planc_2             96         4.34
130      relmeanelev_16             96         4.34
111            planc_32             96         4.34
129           relelev_8             96         4.34
136               sl_16             96         4.34
131       relmeanelev_2             96         4.34
155               tri_2             96         4.34
157               tri_8             96         4.34
158              tsc_16             96         4.34
159               tsc_2             96         4.34
160              tsc_32             96         4.34
161               tsc_4             96         4.34
162               tsc_8             96         4.34
163                 twi             96         4.34
164                vd_4             96         4.34
165                vd_5             96         4.34
166                vd_6             96         4.34
167              vdcn_4             96         4.34
168              vdcn_5             96         4.34
169              vdcn_6             96         4.34
170              vrm_16             96         4.34
171               vrm_2             96         4.34
172              vrm_32             96         4.34
173               vrm_8             96         4.34
156               tri_4             96         4.34
154              tri_16             96         4.34
132      relmeanelev_32             96         4.34
153              tpi_32             96         4.34
133       relmeanelev_4             96         4.34
134       relmeanelev_8             96         4.34
94   pisrdif_2021-10-22             96         4.34
137                sl_2             96         4.34
138               sl_32             96         4.34
139                sl_4             96         4.34
140                sl_8             96         4.34
141        slopeRadians             96         4.34
142                 spi             96         4.34
143       stddevelev_16             96         4.34
144        stddevelev_2             96         4.34
145       stddevelev_32             96         4.34
146        stddevelev_4             96         4.34
147        stddevelev_8             96         4.34
148                 sth             96         4.34
149              swi_10             96         4.34
152               tpi_2             96         4.34
95   pisrdif_2021-11-22             96         4.34
96   pisrdif_2021-12-22             96         4.34
93   pisrdif_2021-09-22             96         4.34
48               maxc_2             96         4.34
29            genelev_2             96         4.34
30           genelev_32             96         4.34
31            genelev_4             96         4.34
32            genelev_8             96         4.34
33           gmrph_r_30             96         4.34
34          gmrph_r_300             96         4.34
35         gmrph_r_3000             96         4.34
37                  hac             96         4.34
38                  hbc             96         4.34
92   pisrdif_2021-08-22             96         4.34
40                hs_cs             96         4.34
41                hs_st             96         4.34
42             longc_16             96         4.34
43              longc_2             96         4.34
44             longc_32             96         4.34
45              longc_4             96         4.34
46              longc_8             96         4.34
28           genelev_16             96         4.34
23       diffmeanelev_8             96         4.34
22       diffmeanelev_4             96         4.34
11              crosc_8             96         4.34
2              aspct_32             96         4.34
3               aspct_4             96         4.34
4               aspct_8             96         4.34
7              crosc_16             96         4.34
8               crosc_2             96         4.34
9              crosc_32             96         4.34
10              crosc_4             96         4.34
12                  dah             96         4.34
21      diffmeanelev_32             96         4.34
14       devmeanelev_16             96         4.34
15        devmeanelev_2             96         4.34
16       devmeanelev_32             96         4.34
17        devmeanelev_4             96         4.34
18        devmeanelev_8             96         4.34
19      diffmeanelev_16             96         4.34
20       diffmeanelev_2             96         4.34
47              maxc_16             96         4.34
39                   hn             96         4.34
49              maxc_32             96         4.34
83          perctelev_4             96         4.34
72          morpfeat_32             96         4.34
73           morpfeat_4             96         4.34
74           morpfeat_8             96         4.34
75                  msp             96         4.34
80                no_32             96         4.34
81         perctelev_16             96         4.34
50               maxc_4             96         4.34
84          perctelev_8             96         4.34
70          morpfeat_16             96         4.34
85   pisrdif_2021-01-22             96         4.34
86   pisrdif_2021-02-22             96         4.34
87   pisrdif_2021-03-22             96         4.34
88   pisrdif_2021-04-22             96         4.34
89   pisrdif_2021-05-22             96         4.34
90   pisrdif_2021-06-22             96         4.34
91   pisrdif_2021-07-22             96         4.34
71           morpfeat_2             96         4.34
82         perctelev_32             96         4.34
68            minelev_8             96         4.34
58           meanelev_8             96         4.34
67            minelev_4             96         4.34
51               maxc_8             96         4.34
52              mbi_001             96         4.34
53               mbi_01             96         4.34
54                mbi_1             96         4.34
56           meanelev_2             96         4.34
57          meanelev_32             96         4.34
55          meanelev_16             96         4.34
59              minc_16             96         4.34
63               minc_8             96         4.34
60               minc_2             96         4.34
65            minelev_2             96         4.34
64           minelev_16             96         4.34
66           minelev_32             96         4.34
62               minc_4             96         4.34
61              minc_32             96         4.34
183            wc_bio10             81         3.67
191            wc_bio18             81         3.67
190            wc_bio17             81         3.67
189            wc_bio16             81         3.67
188            wc_bio15             81         3.67
187            wc_bio14             81         3.67
186            wc_bio13             81         3.67
185            wc_bio12             81         3.67
184            wc_bio11             81         3.67
174            wc_bio01             81         3.67
182            wc_bio09             81         3.67
181            wc_bio08             81         3.67
180            wc_bio07             81         3.67
179            wc_bio06             81         3.67
178            wc_bio05             81         3.67
177            wc_bio04             81         3.67
176            wc_bio03             81         3.67
175            wc_bio02             81         3.67
192            wc_bio19             81         3.67
```

Features with complete data: 0

---

# ================================================================================

# CLASS DISTRIBUTION

# ================================================================================

Burn Severity Class Distribution:
  moderate    : 1021 ( 46.2%)
  low         :  670 ( 30.3%)
  high        :  407 ( 18.4%)
  unburned    :  112 (  5.1%)

Distribution by Fire:

```text
SBS                       high  low  moderate  unburned
Fire_year                                              
alisal_2021                  0    0         0         1
apple_2020                  18   13        27         3
atlas_2017                   0    3         4         2
bald_2014                    2    6         8         4
blueridge_2020               0    2         0         0
bond_2020                    2    0         5         0
borel_2024                   0    0         2         0
briceburg_2019               0    0         1         0
caldor_2021                 57   53        76         5
camp_2018                    1    3         3         0
canyonII_2017                1    5         8         0
carmel_2020                  0    0         1         0
carr_2018                   17   21        57         1
castle_2020                  7    9        22         3
creek_2020                  51   48        52        21
czulightningcomplex_2020     6   28        24         4
day_2014                     2    5         4         0
detwiler_2017                0    0         1         0
dixie_2021                  67  206       351        54
donnell_2018                17    5        20         0
eiler_2014                   3    6        12         1
eldorado_2020               18   22        26         0
emerald_2022                 0    1         1         0
franklin_2024                0    4         2         0
french_2021                  0    3         4         0
getty_2019                   0    1         2         0
glass_2020                   4    5        13         0
hennessey_2020               0    6        33         0
hill_2018                    0    8         1         0
kincade_2019                 2    1         4         0
lake_2024                    0    4         8         0
line_2024                   40   15        29         4
loma_2016                    5    0         3         0
meyers_2020                  0    2         3         0
nuns_2017                    3    9        12         2
oak_2022                    25   36        29         1
palisades_2025               0    3         8         0
park_2024                    1    3        19         0
rim_2013                    31   36        44         2
river_2020                   0    0         5         0
rye_2017                     0    5         1         2
scucomplex_2020              0    7         4         1
silverado_2020               0    3         1         0
skirball_2017                1    1         2         0
sugar_2021                  19   47        33         0
tenaja_2019                  0    1         2         0
tubbs_2017                   1    8        10         0
valley_2020                  2   11         8         0
walbridge_2020               1    8        24         1
woolsey_2018                 3    7        12         0
```

---

# ================================================================================

# PREPARING DATA FOR FEATURE IMPORTANCE ANALYSIS

# ================================================================================

Rows with complete feature data: 1547 (70.0%)

Classes: `['high' 'low' 'moderate' 'unburned']`
Feature matrix shape: (1547, 193)

Train set: 1237 samples
Test set: 310 samples

---

# ================================================================================

# RANDOM FOREST FEATURE IMPORTANCE

# ================================================================================

Training Random Forest...

Random Forest Test Accuracy: 0.2677
Macro F1 Score: 0.2630

Top 30 Features by Random Forest Importance (MDI):

```text
            feature  rf_importance
25             dnbr       0.046186
26            dndbi       0.046064
13             dbsi       0.045763
24           dmndwi       0.035422
36        greenBand       0.031017
79          nirBand       0.030797
27            dndvi       0.029173
69            mndwi       0.025817
5          blueBand       0.024149
151       swir2Band       0.022061
76              nbr       0.021654
150       swir1Band       0.020839
78             ndvi       0.020733
124         redBand       0.019412
6               bsi       0.016510
109        planc_16       0.015675
77             ndbi       0.015048
17    devmeanelev_4       0.013037
133   relmeanelev_4       0.012183
166            vd_6       0.011747
113         planc_8       0.011321
20   diffmeanelev_2       0.010642
111        planc_32       0.010594
2          aspct_32       0.009565
12              dah       0.009110
22   diffmeanelev_4       0.008643
132  relmeanelev_32       0.008520
114            po_2       0.008466
1           aspct_2       0.008161
135           rlp_6       0.007817
```

---

# ================================================================================

# XGBOOST FEATURE IMPORTANCE

# ================================================================================

Training XGBoost...

XGBoost Test Accuracy: 0.2710
Macro F1 Score: 0.2658

Top 30 Features by XGBoost Importance (Gain):

```text
                feature  xgb_importance
109            planc_16        0.082906
111            planc_32        0.044587
62               minc_4        0.041966
179            wc_bio06        0.029152
113             planc_8        0.025705
166                vd_6        0.020173
125          relelev_16        0.018583
171               vrm_2        0.018446
100  pisrdir_2021-04-22        0.018362
102  pisrdir_2021-06-22        0.016232
153              tpi_32        0.016225
75                  msp        0.014578
169              vdcn_6        0.012214
99   pisrdir_2021-03-22        0.012017
60               minc_2        0.011617
142                 spi        0.011311
116            profc_16        0.011155
181            wc_bio08        0.011049
46              longc_8        0.010892
135               rlp_6        0.010683
180            wc_bio07        0.010620
112             planc_4        0.010501
81         perctelev_16        0.010349
126           relelev_2        0.010344
119             profc_4        0.010338
183            wc_bio10        0.010197
51               maxc_8        0.009833
110             planc_2        0.009152
170              vrm_16        0.009071
123              rdgh_6        0.009017
```

---

# ================================================================================

# PERMUTATION IMPORTANCE (using XGBoost)

# ================================================================================

Computing permutation importance (this may take a moment)...

Top 30 Features by Permutation Importance:

```text
                feature  perm_importance_mean  perm_importance_std
25                 dnbr              0.010161             0.010449
191            wc_bio18              0.008227             0.000000
179            wc_bio06              0.008076             0.003436
183            wc_bio10              0.007802             0.002555
7              crosc_16              0.006140             0.002800
166                vd_6              0.005746             0.002054
152               tpi_2              0.005537             0.003833
54                mbi_1              0.004139             0.001326
151           swir2Band              0.001582             0.005826
36            greenBand              0.001106             0.005580
1               aspct_2              0.001098             0.001822
52              mbi_001              0.000811             0.001653
14       devmeanelev_16              0.000579             0.006598
105  pisrdir_2021-09-22              0.000000             0.000000
131       relmeanelev_2              0.000000             0.000000
126           relelev_2              0.000000             0.000000
101  pisrdir_2021-05-22              0.000000             0.000000
127          relelev_32              0.000000             0.000000
128           relelev_4              0.000000             0.000000
129           relelev_8              0.000000             0.000000
130      relmeanelev_16              0.000000             0.000000
100  pisrdir_2021-04-22              0.000000             0.000000
99   pisrdir_2021-03-22              0.000000             0.000000
132      relmeanelev_32              0.000000             0.000000
123              rdgh_6              0.000000             0.000000
133       relmeanelev_4              0.000000             0.000000
134       relmeanelev_8              0.000000             0.000000
136               sl_16              0.000000             0.000000
137                sl_2              0.000000             0.000000
138               sl_32              0.000000             0.000000
```

---

# ================================================================================

# AGGREGATE FEATURE RANKINGS

# ================================================================================

Top 50 Features by Combined Score:

```text
                feature  rf_importance  xgb_importance  perm_importance_mean   avg_rank  combined_score
0                  dnbr       0.046186        0.004998              0.010161  25.333333        0.686764
135            wc_bio06       0.001669        0.029152              0.008076  47.666667        0.394185
19                 vd_6       0.011747        0.020173              0.005746  10.666667        0.354377
139            wc_bio18       0.001548        0.006728              0.008227  62.666667        0.308130
175            wc_bio10       0.000765        0.010197              0.007802  68.666667        0.302456
4             greenBand       0.031017        0.003964              0.001106  36.666667        0.276076
22             planc_32       0.010594        0.044587              0.000000  38.000000        0.255729
45             crosc_16       0.004787        0.001905              0.006140  56.333333        0.243643
9             swir2Band       0.022061        0.005545              0.001582  28.000000        0.233405
144               tpi_2       0.001449        0.006549              0.005537  67.000000        0.218422
20              planc_8       0.011321        0.025705              0.000000  38.333333        0.185057
121              minc_4       0.001936        0.041966              0.000000  71.333333        0.182700
131               mbi_1       0.001709        0.007705              0.004139  58.666667        0.179108
10                  nbr       0.021654        0.003356             -0.000979  96.666667        0.137670
17        devmeanelev_4       0.013037        0.004487              0.000000  63.333333        0.112132
13              redBand       0.019412        0.006520             -0.001654  80.000000        0.112039
28              aspct_2       0.008161        0.001516              0.001098  54.333333        0.101015
68                vrm_2       0.003364        0.018446              0.000000  55.333333        0.098443
31              planc_4       0.007765        0.010501              0.000000  47.666667        0.098262
26       relmeanelev_32       0.008520        0.008907              0.000000  49.000000        0.097301
50   pisrdir_2021-06-22       0.004307        0.016232              0.000000  50.000000        0.096346
30              planc_2       0.007814        0.009152              0.000000  49.333333        0.093189
5               nirBand       0.030797        0.006262             -0.004812  80.333333        0.089584
111  pisrdir_2021-04-22       0.002139        0.018362              0.000000  70.000000        0.089265
21       diffmeanelev_2       0.010642        0.002902              0.000000  73.333333        0.088478
18        relmeanelev_4       0.012183        0.000000              0.000000  90.000000        0.087927
159          relelev_16       0.001134        0.018583              0.000000  85.333333        0.082900
51             profc_16       0.004304        0.011155              0.000000  52.666667        0.075915
46             wc_bio07       0.004581        0.010620              0.000000  52.333333        0.075758
59               minc_2       0.003857        0.011617              0.000000  54.666667        0.074544
24                  dah       0.009110        0.002121              0.000000  76.333333        0.074277
61                  msp       0.003655        0.014578             -0.000374  80.666667        0.072722
25       diffmeanelev_4       0.008643        0.002498              0.000000  75.000000        0.072422
23             aspct_32       0.009565        0.000724              0.000000  80.333333        0.071942
32       devmeanelev_32       0.007759        0.003788              0.000000  74.000000        0.071226
40             wc_bio02       0.005090        0.007830              0.000000  55.000000        0.068220
62              profc_4       0.003640        0.010338              0.000000  59.000000        0.067834
34                   hn       0.006651        0.004231              0.000000  70.666667        0.065011
27                 po_2       0.008466        0.000000              0.000000  93.000000        0.061098
70               rdgh_6       0.003337        0.009017              0.000000  63.333333        0.060337
57               tsc_16       0.003975        0.007612              0.000000  61.333333        0.059292
43        devmeanelev_8       0.004888        0.005926              0.000000  64.666667        0.059104
49       relmeanelev_16       0.004352        0.006854              0.000000  61.000000        0.058967
156  pisrdir_2021-03-22       0.001181        0.012017              0.000000  86.666667        0.056838
77       devmeanelev_16       0.003093        0.003854              0.000579  63.000000        0.056833
180             mbi_001       0.000672        0.006096              0.000811  83.333333        0.055968
145             longc_8       0.001441        0.010892              0.000000  84.666667        0.054188
33        relmeanelev_8       0.007464        0.000000              0.000000  95.000000        0.053866
87   pisrdir_2021-12-22       0.002777        0.008396              0.000000  70.000000        0.053799
160            wc_bio08       0.001110        0.011049              0.000000  89.333333        0.052436
```

---

# ================================================================================

# FEATURE IMPORTANCE BY CATEGORY

# ================================================================================

Feature Category Importance Summary:

```text
                     mean_score  max_score  feature_count  mean_rank
category                                                            
Spectral Bands           0.1053     0.2761              6    70.6667
Climate (WorldClim)      0.0840     0.3942             19    88.3860
Curvature                0.0569     0.2557             35    88.8667
Slope/Aspect             0.0481     0.1010              6    91.9444
Terrain Indices          0.0425     0.2184             15   103.6222
Solar Radiation          0.0312     0.0963             24   105.5000
Elevation                0.0274     0.1121             43    99.9690
Hydrology                0.0251     0.3544             24    97.9444
Geomorphology            0.0218     0.0743             11   123.6667
Differenced Indices     -0.1900     0.6868              5    79.0000
Spectral Indices        -0.4137     0.1377              5    92.9333
```

Top 5 Features per Category:

**Spectral Bands:**
  greenBand                      | Score: 0.2761 | Rank: 36.7
  swir2Band                      | Score: 0.2334 | Rank: 28.0
  redBand                        | Score: 0.1120 | Rank: 80.0
  nirBand                        | Score: 0.0896 | Rank: 80.3
  swir1Band                      | Score: -0.0128 | Rank: 94.3

**Climate (WorldClim):**
  wc_bio06                       | Score: 0.3942 | Rank: 47.7
  wc_bio18                       | Score: 0.3081 | Rank: 62.7
  wc_bio10                       | Score: 0.3025 | Rank: 68.7
  wc_bio07                       | Score: 0.0758 | Rank: 52.3
  wc_bio02                       | Score: 0.0682 | Rank: 55.0

**Curvature:**
  planc_32                       | Score: 0.2557 | Rank: 38.0
  crosc_16                       | Score: 0.2436 | Rank: 56.3
  planc_8                        | Score: 0.1851 | Rank: 38.3
  minc_4                         | Score: 0.1827 | Rank: 71.3
  planc_4                        | Score: 0.0983 | Rank: 47.7

**Slope/Aspect:**
  aspct_2                        | Score: 0.1010 | Rank: 54.3
  aspct_32                       | Score: 0.0719 | Rank: 80.3
  aspct_8                        | Score: 0.0459 | Rank: 80.7
  aspct_16                       | Score: 0.0328 | Rank: 112.0
  aspct_4                        | Score: 0.0187 | Rank: 108.7

**Terrain Indices:**
  tpi_2                          | Score: 0.2184 | Rank: 67.0
  vrm_2                          | Score: 0.0984 | Rank: 55.3
  rdgh_6                         | Score: 0.0603 | Rank: 63.3
  spi                            | Score: 0.0504 | Rank: 95.0
  twi                            | Score: 0.0453 | Rank: 79.3

**Solar Radiation:**
  pisrdir_2021-06-22             | Score: 0.0963 | Rank: 50.0
  pisrdir_2021-04-22             | Score: 0.0893 | Rank: 70.0
  pisrdir_2021-03-22             | Score: 0.0568 | Rank: 86.7
  pisrdir_2021-12-22             | Score: 0.0538 | Rank: 70.0
  pisrdir_2021-02-22             | Score: 0.0395 | Rank: 86.0

**Elevation:**
  devmeanelev_4                  | Score: 0.1121 | Rank: 63.3
  relmeanelev_32                 | Score: 0.0973 | Rank: 49.0
  diffmeanelev_2                 | Score: 0.0885 | Rank: 73.3
  relmeanelev_4                  | Score: 0.0879 | Rank: 90.0
  relelev_16                     | Score: 0.0829 | Rank: 85.3

**Hydrology:**
  vd_6                           | Score: 0.3544 | Rank: 10.7
  mbi_1                          | Score: 0.1791 | Rank: 58.7
  hn                             | Score: 0.0650 | Rank: 70.7
  po_2                           | Score: 0.0611 | Rank: 93.0
  mbi_001                        | Score: 0.0560 | Rank: 83.3

**Geomorphology:**
  dah                            | Score: 0.0743 | Rank: 76.3
  msp                            | Score: 0.0727 | Rank: 80.7
  sth                            | Score: 0.0349 | Rank: 98.7
  gmrph_r_30                     | Score: 0.0304 | Rank: 103.3
  morpfeat_32                    | Score: 0.0217 | Rank: 119.3

**Differenced Indices:**
  dnbr                           | Score: 0.6868 | Rank: 25.3
  dmndwi                         | Score: 0.0163 | Rank: 97.3
  dndbi                          | Score: -0.0283 | Rank: 82.7
  dndvi                          | Score: -0.1719 | Rank: 97.3
  dbsi                           | Score: -1.4530 | Rank: 92.3

**Spectral Indices:**
  nbr                            | Score: 0.1377 | Rank: 96.7
  mndwi                          | Score: -0.0140 | Rank: 80.7
  ndvi                           | Score: -0.1235 | Rank: 87.0
  ndbi                           | Score: -0.7846 | Rank: 91.7
  bsi                            | Score: -1.2843 | Rank: 108.7

Full results saved to: `results/feature_importance_results.csv`
Top 50 features saved to: `results/top_50_features.txt`

---

# ================================================================================

# GENERATING VISUALIZATIONS

# ================================================================================

Saved: `feature_importance_comparison.png`
Saved: `category_importance.png`
Saved: `feature_importance_heatmap.png`

---

# ================================================================================

# RECOMMENDATIONS FOR FEATURE SELECTION

# ================================================================================

High importance features (score > 0.3): 5
Medium importance features (0.1 < score <= 0.3): 12

Missing Data in Top 30 Features:

```text
  dnbr                          :  520 missing (23.5%)
  wc_bio06                      :   81 missing (3.7%)
  vd_6                          :   96 missing (4.3%)
  wc_bio18                      :   81 missing (3.7%)
  wc_bio10                      :   81 missing (3.7%)
  greenBand                     :  520 missing (23.5%)
  planc_32                      :   96 missing (4.3%)
  crosc_16                      :   96 missing (4.3%)
  swir2Band                     :  520 missing (23.5%)
  tpi_2                         :   96 missing (4.3%)
  planc_8                       :   96 missing (4.3%)
  minc_4                        :   96 missing (4.3%)
  mbi_1                         :   96 missing (4.3%)
  nbr                           :  520 missing (23.5%)
  devmeanelev_4                 :   96 missing (4.3%)
  redBand                       :  520 missing (23.5%)
  aspct_2                       :   96 missing (4.3%)
  vrm_2                         :   96 missing (4.3%)
  planc_4                       :   96 missing (4.3%)
  relmeanelev_32                :   96 missing (4.3%)
  pisrdir_2021-06-22            :   96 missing (4.3%)
  planc_2                       :   96 missing (4.3%)
  nirBand                       :  520 missing (23.5%)
  pisrdir_2021-04-22            :   96 missing (4.3%)
  diffmeanelev_2                :   96 missing (4.3%)
  relmeanelev_4                 :   96 missing (4.3%)
  relelev_16                    :   96 missing (4.3%)
  profc_16                      :   96 missing (4.3%)
  wc_bio07                      :   81 missing (3.7%)
  minc_2                        :   96 missing (4.3%)
```

---

# ================================================================================

# SUGGESTED FEATURE SETS FOR MODELING

# ================================================================================

1. **MINIMAL SET (Top 15 features):**

   1. dnbr
   2. wc_bio06
   3. vd_6
   4. wc_bio18
   5. wc_bio10
   6. greenBand
   7. planc_32
   8. crosc_16
   9. swir2Band
   10. tpi_2
   11. planc_8
   12. minc_4
   13. mbi_1
   14. nbr
   15. devmeanelev_4

2. **RECOMMENDED SET (Top 30 features):**

   1. dnbr
   2. wc_bio06
   3. vd_6
   4. wc_bio18
   5. wc_bio10
   6. greenBand
   7. planc_32
   8. crosc_16
   9. swir2Band
   10. tpi_2
   11. planc_8
   12. minc_4
   13. mbi_1
   14. nbr
   15. devmeanelev_4
   16. redBand
   17. aspct_2
   18. vrm_2
   19. planc_4
   20. relmeanelev_32
   21. pisrdir_2021-06-22
   22. planc_2
   23. nirBand
   24. pisrdir_2021-04-22
   25. diffmeanelev_2
   26. relmeanelev_4
   27. relelev_16
   28. profc_16
   29. wc_bio07
   30. minc_2

3. **EXTENDED SET (Top 50 features):**

   1. dnbr
   2. wc_bio06
   3. vd_6
   4. wc_bio18
   5. wc_bio10
   6. greenBand
   7. planc_32
   8. crosc_16
   9. swir2Band
   10. tpi_2
   11. planc_8
   12. minc_4
   13. mbi_1
   14. nbr
   15. devmeanelev_4
   16. redBand
   17. aspct_2
   18. vrm_2
   19. planc_4
   20. relmeanelev_32
   21. pisrdir_2021-06-22
   22. planc_2
   23. nirBand
   24. pisrdir_2021-04-22
   25. diffmeanelev_2
   26. relmeanelev_4
   27. relelev_16
   28. profc_16
   29. wc_bio07
   30. minc_2
   31. dah
   32. msp
   33. diffmeanelev_4
   34. aspct_32
   35. devmeanelev_32
   36. wc_bio02
   37. profc_4
   38. hn
   39. po_2
   40. rdgh_6
   41. tsc_16
   42. devmeanelev_8
   43. relmeanelev_16
   44. pisrdir_2021-03-22
   45. devmeanelev_16
   46. mbi_001
   47. longc_8
   48. relmeanelev_8
   49. pisrdir_2021-12-22
   50. wc_bio08

FEATURES IN TOP 50 THAT NEED IMPUTATION:

```text
  dnbr                          :  520 missing values
  wc_bio06                      :   81 missing values
  vd_6                          :   96 missing values
  wc_bio18                      :   81 missing values
  wc_bio10                      :   81 missing values
  greenBand                     :  520 missing values
  planc_32                      :   96 missing values
  crosc_16                      :   96 missing values
  swir2Band                     :  520 missing values
  tpi_2                         :   96 missing values
  planc_8                       :   96 missing values
  minc_4                        :   96 missing values
  mbi_1                         :   96 missing values
  nbr                           :  520 missing values
  devmeanelev_4                 :   96 missing values
  redBand                       :  520 missing values
  aspct_2                       :   96 missing values
  vrm_2                         :   96 missing values
  planc_4                       :   96 missing values
  relmeanelev_32                :   96 missing values
  pisrdir_2021-06-22            :   96 missing values
  planc_2                       :   96 missing values
  nirBand                       :  520 missing values
  pisrdir_2021-04-22            :   96 missing values
  diffmeanelev_2                :   96 missing values
  relmeanelev_4                 :   96 missing values
  relelev_16                    :   96 missing values
  profc_16                      :   96 missing values
  wc_bio07                      :   81 missing values
  minc_2                        :   96 missing values
  dah                           :   96 missing values
  msp                           :   96 missing values
  diffmeanelev_4                :   96 missing values
  aspct_32                      :   96 missing values
  devmeanelev_32                :   96 missing values
  wc_bio02                      :   81 missing values
  profc_4                       :   96 missing values
  hn                            :   96 missing values
  po_2                          :   96 missing values
  rdgh_6                        :   96 missing values
  tsc_16                        :   96 missing values
  devmeanelev_8                 :   96 missing values
  relmeanelev_16                :   96 missing values
  pisrdir_2021-03-22            :   96 missing values
  devmeanelev_16                :   96 missing values
  mbi_001                       :   96 missing values
  longc_8                       :   96 missing values
  relmeanelev_8                 :   96 missing values
  pisrdir_2021-12-22            :   96 missing values
  wc_bio08                      :   81 missing values
```

---

# ================================================================================

# ANALYSIS COMPLETE

# ================================================================================
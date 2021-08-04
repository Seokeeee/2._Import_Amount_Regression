# Project : Import Amount Regression

<img width="80%" src="/images/01_Intro.jpg"/>

## 1. Introduction

최근, 북미와 일부 유럽 국가에서 일어난 반세계화 정서는 세계화(Globalization)에 대한 논의와 비즈니스 리더들의 전략에 새로운 고민을 던지고 있습니다.

이 고민 중 가장 뜨거운 감자는 바로 무역(Trade)입니다.

특히, 오늘날의 대한민국은 국민총소득(GNI)과 국민총생산(GDP)에 기여하는 무역의 의존도가 매우 높은 나라입니다.

2006년을 기점으로 2019년까지 대한민국의 무역의존도는 매년  60%이상을 기록하고 있습니다. 

이러한 이유로 무역에 대한 관리와 대비는 대한민국 입장에서 매우 중요한 과제 중 하나입니다.

오늘은 특정 품목과 특정 국가의 데이터를 활용하여 차년도에 해당 국가가 해당 품목으로 대한민국으로부터 얼만큼 수입했는지를 예측하는 과제를 신경망을 이용해 수행합니다.

또한, 이를 모델링 관점과 데이터 관점에서 살펴봅니다.



Keyword : Regression, Artificial Intelligence, Neural Network, Auto-Machine Learning



## 2. Analysis

#### 2.1. Data Overview

데이터 : KOTRA 무역 데이터

수집 기간 : 2017년 ~ 2018년, 1년 

수집 내용 : 제9회 공공데이터 활용 빅데이터 분석공모전 데이터

수집 방법 : 공모전 홈페이지

데이터 설명 :

| 컬럼명                  | 설명                                                         | 단위           |
| ----------------------- | ------------------------------------------------------------ | -------------- |
| UNC_YEAR                | 기준연도                                                     | YYYY           |
| HSCD                    | HS Code (품목코드)                                           | 6자리 숫자코드 |
| COUNTRYCD               | ISO 국가코드                                                 | 숫자코드       |
| COUNTRYNM               | 영문 국가명                                                  | Character      |
| TRADE_COUNTRYCD         | 해당 연도 해당 국가의 전체 품목 수입금액                     | US$            |
| TRADE_HSCD              | 해당 연도 해당 품목의 전세계 총 수입금액                     | US$            |
| TARIFF_AVG              | 해당 국가에서 해당 품목에 적용되는 평균 관세율               | %              |
| SNDIST                  | 해당 국가와 수입 국가 간 평균 거리                           | km             |
| NY_GDP_MKTP_CD          | GDP                                                          | US$            |
| NY_GDP_MKTP_CD_1Y       | 이전년도 GDP                                                 | US$            |
| SP_POP_TOTL             | 인구 (연중 추정치)                                           | 명             |
| PA_NUS_FCRF             | 공식 환율 (미국 달러에 대한 현지 통화 단위, 월평균을 기준으로 한 연평균) | US$            |
| IC_BUS_EASE_DFRN_DB     | 비즈니스 용이성 점수                                         | 점수 (0~100)   |
| KMDIST                  | 해당 국가와 한국과의 거리                                    | km             |
| TRADE_HSCD_COUNTRYCD    | 해당 연도 해당 국가의 해당 품목 수입금액                     | US$            |
| KR_TRADE_HSCD_COUNTRYCD | 내년 해당 국가가 해당 품목을 한국으로부터 수입한 금액        | US$            |



#### 2.2. Modeling Issue

이번 과제는 차년도에 해당 국가가 해당 품목을 대한민국으로부터 수입한 금액을 예측하는 문제이며, 일반적으로 회귀 모델(Regression Model)을 이용합니다.

전통적인 회귀 모델에는 다양한 기법과 가정이 이용되지만 본 포스팅은 신경망(Neural Network)를 이용하였습니다.



<img width="80%" src="/images/02_Neural_Network.png"/>

다른 회귀 모델과 달리 신경망은 정의된 과제를 해결하기위해 사용된 특징들(Features)을 보다 풍부하고 유연하게 표현(Representation)할 수 있다는 장점이 있습니다.

이러한 장점은 이번에 분석할 데이터의 특징과도 상당한 연관이 있습니다. 

본 과제의 가장 직관적인 특징은 단연코 "TRADE_" 계열의 특징들일 것입니다.

하지만, 본 데이터는 "TRADE_"에 큰 영향을 주는 또다른 특징들이 많이 포함되어 있습니다.

이러한 특징들 또한 회귀 모델에 적극적으로 반영할 수 있다면, 과제 해결에 보다 큰 도움을 줄 수 있습니다.

따라서 본 프로젝트의 모델은 신경망을 활용하였습니다.



#### 2.3. EDA(Exploratory Data Analysis)

모델링에 앞서 먼저, 데이터에 대해 간단히 살펴봅니다.

컬럼별 Null값 개수

```python
UNC_YEAR                      0
HSCD                          0
COUNTRYCD                     0
COUNTRYNM                     0
TRADE_COUNTRYCD               0
TRADE_HSCD                    0
TARIFF_AVG                  129
SNDIST                       22
NY_GDP_MKTP_CD                0
NY_GDP_MKTP_CD_1Y             0
SP_POP_TOTL                   0
PA_NUS_FCRF                3488
IC_BUS_EASE_DFRN_DB           0
KMDIST                        0
TRADE_HSCD_COUNTRYCD         21
KR_TRADE_HSCD_COUNTRYCD       0
```

Null값이 포함된 데이터를 처리하지 않고 회귀 모델 학습에 이용한다면, 학습 자체에 오류를 발생시키거나 잘못된 학습 결과를 야기할 수 있습니다. 

총 3,606개의 null값이 포함된 데이터를 삭제하여 총 17,583개의 데이터를 이용합니다.

 또한, UNC_YEAR, HSCD, COUNTRYCD, COUNTRYNM는 COUNTRYNM를 제외하곤 정수값을 갖지만 엄밀히 말하면 수치형을 뜻하진 않습니다.

최근에는 이러한 데이터도 최대한 활용하는 연구들이 진행되고 있지만 본 프로젝트에서는 사용하지 않습니다.

간단한 통계학적 자료를 살펴보면 다음과 같습니다.

<img width="80%" src="/images/03_EDA.png"/>

|       | TRADE_COUNTRYCD |   TRADE_HSCD |   TARIFF_AVG |       SNDIST | NY_GDP_MKTP_CD | NY_GDP_MKTP_CD_1Y |  SP_POP_TOTL |  PA_NUS_FCRF | IC_BUS_EASE_DFRN_DB |       KMDIST | TRADE_HSCD_COUNTRYCD | KR_TRADE_HSCD_COUNTRYCD |
| :---- | --------------: | -----------: | -----------: | -----------: | -------------: | ----------------: | -----------: | -----------: | ------------------: | -----------: | -------------------: | ----------------------- |
| count |    1.758300e+04 | 1.758300e+04 | 17583.000000 | 17583.000000 |   1.758300e+04 |      1.758300e+04 | 1.758300e+04 | 17583.000000 |        17583.000000 | 17583.000000 |         1.758300e+04 | 1.758300e+04            |
| mean  |    3.084354e+11 | 1.449538e+10 |     3.914770 |  6756.914745 |   1.650877e+12 |      1.552090e+12 | 1.384541e+08 |  1992.788134 |           69.675625 |  7770.034252 |         2.683105e+08 | 1.991493e+07            |
| std   |    4.833259e+11 | 3.600955e+10 |     9.384518 |  2558.589304 |   3.725238e+12 |      3.533778e+12 | 3.085904e+08 |  6564.730453 |           11.195796 |  4409.078529 |         2.061493e+09 | 5.273917e+08            |
| min   |    4.337305e+09 | 2.436821e+08 |     0.000000 |  1172.047241 |   1.142576e+10 |      1.118673e+10 | 3.113779e+06 |     0.303350 |           42.671390 |   955.651062 |         7.630000e+02 | 0.000000e+00            |
| 25%   |    5.161228e+10 | 2.446361e+09 |     0.000000 |  4621.086469 |   2.186289e+11 |      1.962721e+11 | 1.059444e+07 |     3.191389 |           59.969240 |  4614.067383 |         7.871426e+06 | 4.769350e+04            |
| 50%   |    1.628989e+11 | 4.837572e+09 |     0.000000 |  6457.334844 |   3.856055e+11 |      3.570451e+11 | 3.654332e+07 |     7.793250 |           71.581850 |  7730.766602 |         3.101242e+07 | 3.510270e+05            |
| 75%   |    3.277097e+11 | 1.174054e+10 |     5.000000 |  8119.394952 |   1.329188e+12 |      1.208847e+12 | 1.051733e+08 |   110.973017 |           78.272910 | 10293.836910 |         1.114722e+08 | 2.404007e+06            |
| max   |    2.405277e+12 | 3.473137e+11 |   515.000000 | 15134.164110 |   1.951935e+13 |      1.871496e+13 | 1.386395e+09 | 33226.298150 |           87.166330 | 18375.181640 |         1.130730e+11 | 6.369533e+10            |

특히, 단위가 화폐인 특징들은 각 샘플들 사이의 차이가 매우 큰 것을 확인할 수 있습니다.

이는 단위가 큰 무역 데이터으로부터 나타난 특징이며, 특별한 처리를 하지 않고 사용할 경우 큰 편차로 인해 올바른 매개변수(Parameter) 학습에 악영향을 끼칠 수 있습니다.

본 프로젝트에서는 이를 Z-Score Normalization을 이용하여 각 샘플들의 값 자체의 편차를 줄이되, 이 값이 갖는 의미는 내포하도록 표현하였습니다. 

Z-Score Normalization된 샘플

|       | TRADE_COUNTRYCD | TRADE_HSCD | TARIFF_AVG |    SNDIST | NY_GDP_MKTP_CD | NY_GDP_MKTP_CD_1Y | SP_POP_TOTL | PA_NUS_FCRF | IC_BUS_EASE_DFRN_DB |    KMDIST | TRADE_HSCD_COUNTRYCD |
| :---- | --------------: | ---------: | ---------: | --------: | -------------: | ----------------: | ----------: | ----------: | ------------------: | --------: | -------------------- |
| 18530 |        0.751393 |  -0.231386 |  -0.406374 |  0.376368 |       0.866172 |          0.957343 |   -0.037765 |   -0.289085 |            0.746224 | -1.496824 | -0.005331            |
| 2996  |        0.280441 |  -0.139108 |   0.624609 | -0.557876 |       0.270207 |          0.211434 |    3.883598 |   -0.296198 |           -1.233437 | -0.692018 | -0.072301            |
| 1383  |       -0.360549 |  -0.060095 |   0.109117 | -0.624448 |      -0.258483 |         -0.256896 |   -0.340915 |   -0.305477 |           -0.916371 | -0.040026 | -0.109187            |
| 11077 |       -0.078014 |   2.399898 |   0.109117 | -0.669980 |      -0.340035 |         -0.338616 |   -0.417318 |   -0.305489 |            0.693746 | -0.187521 | 0.090164             |
| 16573 |       -0.556016 |  -0.166833 |  -0.406374 |  2.929581 |      -0.388536 |         -0.386538 |   -0.432505 |   -0.305832 |            1.564759 |  0.582825 | -0.121820            |

 

선형 모델에 학습할 모델은 Train : Validation : Test = 0.6 : 0.2 : 0.2 으로 구분하며 최종 Reporting은 Test 데이터에 대하여 진행하였습니다.



## 3. Modeling-driven Approach

#### 3.1. Baseline_01

Baseline_01 모델 개요

```python
Baseline_01 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation = 'relu', input_shape=Data_Shape),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam()
```

Baseline_01 학습 결과

<img width="80%" src="/images/04_BL_01.png"/>

Reporting

<img width="80%" src="/images/05_Performance_01.png"/>



#### 3.2. Baseline_02

Baseline_01은 두 개의 은닉층을 사용한 회귀 모델입니다.

은닉층의 깊이가 모델 성능에 어떤 영향을 미치는지 비교하기 위하여 이번 비교 모델은 한 개의 은닉층만을 사용합니다.

Baseline_02 모델 개요

```python
Baseline_02 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation = 'relu', input_shape=Data_Shape),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam()
```

Baseline_02 학습 결과

<img width="80%" src="/images/06_BL_02.png"/>

Baseline_01과 비교하여 Baseline_02의 학습 그래프가 정적인 것을 알 수 있습니다.

실제로 Baseline_01과 Baseline_02는 제대로된 학습을 진행하였지만 해결하고자 하는 과제의 단위가 너무 커, 명확한 식별이 안되는 것을 확인할 수 있습니다.

이것에 대하여 이후에 보다 자세히 논의합니다.

Reporting

<img width="80%" src="/images/07_Performance_02.png"/>

두 가지의 간단한 신경망 기반의 회귀 모델은  위와 같은 성능을 발휘하였습니다.

Baseline_01은 Baseline_02와 비교하여 MSE와 RMSE에서 보다 좋은 성능을 발휘하였지만 MAE에서는 좋지 못한 성능을 보였습니다.



#### 3.3. Baseline_03

위의 두 비교 모델은 신경망 기반의 회귀 모델입니다.

이번엔 신경망 기반의 회귀 모델이 본 프로젝트에 얼마만큼 적합성이 있는지 알아보기위해 전혀 다른 비교모델을 이용하였습니다.

최근 Kaggle 등의 Competition에서 우수한 성능을 보이고 있는 XGBoost 기반의 회귀 모델입니다.

Baseline_02 모델 개요

<img width="80%" src="/images/08_BL_03.png"/>

Reporting

<img width="80%" src="/images/09_Performance_03.png"/>

위의 성능과 비교하였을 때, 신경망 기반의 회귀 모델과 XGBoost 기반의 회귀 모델 각각 취할 수 있는 장.단점이 구분되는 것을 확인할 수 있습니다. 

(본 포스팅은 XGBoost 모델에 대한 자세한 설명을 다루지 않습니다.)



#### 3.4. Baseline_04

앞의 비교 모델들에 사용된 데이터를 다시 한 번 상기하면 본 프로젝트에서 사용된 데이터는 단위와 각 샘플들 사이의 차이가 큰 특징이 있음을 확인하였습니다.

이를 위하여 특징들에 대하여  Z-Score Normalization을 진행하였지만 레이블에 해당하는 KR_TRADE_HSCD_COUNTRYCD는 Normalization이 진행하지 않았습니다.

이는 다른 특징들과 달리 회귀 모델의 최종 결과값은 해석 가능한 수치여야 한다는 것입니다.

달리 말하여, 최종 결과값은 어떤 의미가 있는 정규화된  값이 아닌 US$로 표기할 수 있는 값이여야 한다는 것입니다.

이를 위하여 본 프로젝트는 서로 역함수 관계인 자연지수와 자연로그의 특징을 이용하였습니다.

KR_TRADE_HSCD_COUNTRYCD 로그 변환

```
np.log1p(KR_TRADE_HSCD_COUNTRYCD)
```



Baseline_04 모델 개요

```python
Baseline_04 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation = 'relu', input_shape=Data_Shape),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam()
```



KR_TRADE_HSCD_COUNTRYCD 변환

```
Prediction = Baseline_04.predict(Normed_X_Test)
np.exp(Prediction)
```



Baseline_04 학습 결과

<img width="80%" src="/images/10_BL_04.png"/>

Reporting

<img width="80%" src="/images/11_Performance_04.png"/>

학습 당시의 데이터는 자연로그로 축소한 뒤에 회귀 학습을 진행한 이후에, 예측한 자연로그 값을 역함수의 자연지수 함수에 적용하여 다른 비교 모델과 비교를 진행해보았습니다.

이 방식은 매우 간단하지만 성능이 강력함을 볼 수 있습니다.

앞의 비교 모델과 비교하여 MAE, MSE, RMSE 모든 지표에서 성능을 개선할 수 있었습니다.



#### 3.5. Baseline_05

하지만 여전히 우리의 회귀 모델은 개선할 점이 많습니다.

특히, 신경망 기반의 모델은 다양한 초매개변수(Hyperparameter)가 존재하며 예측 모델의 성능은 초매개변수의 영향을 많이 받습니다.

따라서 이번엔 우리의 회귀 모델이 최고의 성능을 발휘할 수 있도록 초매개변수의 조합을 최적화하였습니다.

이를 위하여 Auto Machine Learning을 이용하였으며, 신경망 유닛의 개수, 활성화 함수의 종류, 학습률에 대하여 최적화된 조합을 탐색하였습니다.



AutoML 개요

```python
class Hypothesis(hp):
    def __init__(self, Data_Shape):
        self.Data_Shape = Data_Shape
        
    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=hp.Int('units', 8, 128, 8, default=8),
                                        activation = hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid'], default='relu'),
                                        input_shape=Data_Shape))
        
        model.add(tf.keras.layers.Dense(1))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]))
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError()])
        
        return model

Models_Hypothesis = Hypothesis(Data_Shape)

Tuner = kt.RandomSearch(
    Models_Hypothesis,
    objective = 'mse',
    max_trials = 50,
    directory = Model_Tuning_path,
    project_name = "Model_Select"
)
```



위의 내용을 토대로 총 50번의 랜덤 탐색을 진행하였으며, 탐색 과정 중 가장 성능이 좋은 10개의 최적화 조합 또는 가설 집합(Hypothesis Set)을  도출하여 다시 한 번 비교 평가를 진행하였습니다.

<img width="80%" src="/images/12_Hypothesis_05.PNG"/>

이 결과로 9번째 가설이 MAE, MSE, RMSE를 종합적으로 고려해보았을 때, 가장 좋은 성능을 보였습니다.

해당 최적화 조합을 이용하여 다음과 같은 비교 모델을 개발하였습니다.



Baseline_05 모델 개요

```python
Baseline_05 = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation = 'sigmoid', input_shape=Data_Shape),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
```



Reporting

<img width="80%" src="/images/13_Performance_05.png"/>

AutoML으로부터 도출된 회귀 모형은 많은 자원과 시간을 요구하지만 성능 개선 또한 확실하였습니다.

특히, MSE와 RMSE에 대하여 많은 개선이 이루어졌습니다.



## 4. Data-driven Approach

#### 4.1. Data Preprocessing

한편, 우리는 2.3. EDA에서 회귀 분석의 용이성을 위하여 Null값이 포함된 3,606개의 데이터를 제거한 후, 회귀 모델링을 수행하였습니다.

이는 21,189개의 전체 데이터 중 약 17%에 해당하는 데이터로 분석 관점에서 매우 큰 손실일 수 있습니다.

이를 개선하기 위하여 이번 절에서는 데이터를 보다 심층적으로 탐색하고 전처리하였습니다.

컬럼별 Null값 개수

```python
UNC_YEAR                      0
HSCD                          0
COUNTRYCD                     0
COUNTRYNM                     0
TRADE_COUNTRYCD               0
TRADE_HSCD                    0
TARIFF_AVG                  129
SNDIST                       22
NY_GDP_MKTP_CD                0
NY_GDP_MKTP_CD_1Y             0
SP_POP_TOTL                   0
PA_NUS_FCRF                3488
IC_BUS_EASE_DFRN_DB           0
KMDIST                        0
TRADE_HSCD_COUNTRYCD         21
KR_TRADE_HSCD_COUNTRYCD       0
```

PA_NUS_FCRF에 대한 처리

다시 한 번, Null값을 살펴보면 가장 많은 Null값을 갖고 있는 열은 PA_NUS_FCRF입니다.

PA_NUS_FCRF는 자국의 화폐 가치를 US$ 단위로 환산한 특징으로 무역과 밀접한 관계가 있는 환율에 대한 특징입니다.

이 Null값을 기준으로 COUNTRYNM를 살펴보니 총 7개의 나라가 검색되었습니다.

Austria, Belgium, France, Germany, Italy, Netherlands, Spain으로 공용 화폐인 유로를 사용하는 나라들이었습니다.

유로는 나라의 화폐가 아닌 연합의 화폐로 이것이 유로 회원국의 환율에 반영되지 못한 채, 데이터에 반영되지 못한 것이 아닌가...라는 추론을 하게 되었습니다.

실제로도 해당 데이터를 만들기 위하여 사용된 세계 은행(World Bank)의 사이트를 방문해 보니 당해에 유로 회원국의 환율이 Null값으로 되어 있었습니다.

이를 데이터에 반영하기 위하여 US$ = 1, UK£ = 0.776977인 점을 감안하여 당해 EU€ 환율을 반영하였습니다.



TARIFF_AVG에 대한 처리

두번 째로 많은 Null값을 차지하는 열인 TARIFF_AVG는 "해당 국가에서 해당 품목에 적용되는 평균 관세율"에 대한 특징입니다.

이 특징에 대하여 보다 심층적인 처리를 하기 위해서는 우선 해당 품목이 무엇인지에 대한 조사를 진행하였습니다.

국제 무역 센터(ITC, International Trade Centre)에 따른 Null값에 해당하는 품목에 대한 내용은 다음과 같습니다.

```python
382499 - Chemical products and preparations of the chemical or allied industries, incl. those consisting of mixtures of natural products, n.e.s.

420232 - Wallets, purses, key-pouches, cigarette-cases, tobacco-pouches and similar articles carried in the pocket or handbag, with outer surface of plastic sheeting or textile materials

847982 - Mixing, kneading, crushing, grinding, screening, sifting, homogenising, emulsifying or stirring machines, n.e.s. (excluding industrial robots)

851690 - Parts of electric water heaters, immersion heaters, space-heating apparatus and soil-heating apparatus, hairdressing apparatus and hand dryers, electro-thermic appliances of a kind used for domestic purposes and electric heating resistors, n.e.s.

852852 - Monitors capable of directly connecting to and designed for use with an automatic data processing machine of heading 8471 (excl. CRT, with TV receiver)

902690 - Parts and accessories for instruments and apparatus for measuring or checking the flow, level, pressure or other variables of liquids or gases, n.e.s.

903290 - Parts and accessories for regulating or controlling instruments and apparatus, n.e.s.

903300 - Parts and accessories for machines, appliances, instruments or other apparatus in chapter 90, specified neither in this chapter nor elsewhere

999999 - Commodities not elsewhere specified
```

위 내용을 살펴보면 주로 어떤 상품군의 부속품이나 혼합물과 같은 관세율을 명확히 정의하기 어려운 품목들이었습니다. 

이와 같은 내용을 정확히 정의하기 위해서는 세계 무역에 대해 정통한 지식이 필요하며, 이는 많은 비용이 요구됩니다.

따라서 본 프로젝트는 다음과 같은 처리를 진행하였습니다.

```python
382499 - 0
420232 - 가장 비슷한 수준의 GDP를 참고
847982 - 가장 비슷한 수준의 GDP를 참고
851690 - 가장 비슷한 수준의 GDP를 참고
852852 - 0
902690 - 가장 비슷한 수준의 GDP를 참고
903290 - 가장 비슷한 수준의 GDP를 참고
903300 - 가장 비슷한 수준의 GDP를 참고
999999 - 0
```



SNDIST와 TRADE_HSCD_COUNTRYCD에 대한 처리

SNDIST는 "해당 국가와  수입 국가 간 평균 거리"에 대한 특징으로 해당 국가의 전체 무역의 평균 거리로 Null값을 대체하였습니다.

TRADE_HSCD_COUNTRYCD는 "해당 연도 해당 국가의 해당 품목 수입금액"에 대한 특징입니다.

다른 특징과 달리 TRADE_HSCD_COUNTRYCD는 Null값을 적절한 값으로 대체하기 까다롭습니다.

해당 품목을 기준으로 모든 나라를 참조하는 방법과 해당 나라를 기준으로 모든 품목을 참조하는 방법 모두 각 샘플 사이의 편차가 매우 커, 큰 편향으로 영향을 줄 수도 있습니다.

따라서 TRADE_HSCD_COUNTRYCD에 Null값을 갖는 샘플은 분석에서 제거하였습니다. 



#### 4.2. Baseline_06

4.1의 과정으로 총 21,189개의 데이터를 활용하여 회귀 모델을 개발합니다.

바뀐 양에 맞춰 다시 한 번 AutoML을 사용하여 초매개변수 튜닝을 진행한 후 다음과 같은 모델을 개발하였습니다.

 

Baseline_06 모델 개요

```python
Baseline_06 = tf.keras.Sequential([
    tf.keras.layers.Dense(96, activation = 'sigmoid', input_shape=Data_Shape),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
```

Baseline_06 학습 결과

<img width="80%" src="/images/14_BL_06.png"/>

Reporting

<img width="80%" src="/images/15_Performance_06.png"/>



## 5. Conculsion

이번 프로젝트는 특정 품목과 특정 국가의 데이터를 활용하여 차년도에 해당 국가가 해당 품목으로 대한민국으로부터 얼만큼 수입했는지를 예측하였습니다.

이를 위하여 신경망을 기반으로 한 회귀 모델에 대하여 다양한 방법으로 회귀 성능을 개선하였습니다.

이를 위하여 본 포스팅은 크게 3가지의 조치를 진행하였습니다.

1. 데이터 자체에 대한 심층적인 탐색을 통하여 충분한 양의 데이터를 확보

2. 효과적인 회귀 학습을 위하여 약간의 트릭(?)을 사용
3. AutoML을 이용하여 최적의 회귀 모델을 탐색

신경망 기반의 회귀 모델 성능을 개선할 수 있는 방법은 위의 방법 이외에 매우 다양하게 있습니다.

이러한 방법을 복합적으로 활용한다면, 더욱 좋은 성능을 기대할 수 있습니다.

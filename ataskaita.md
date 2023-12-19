Gintautas Zenevskis, IFD0

#### Įvadas
Šiame darbe atliekamas šnekamosios kalbos garso įrašų klasifikavimas be giliojo mokymosi.

#### Literatūros apžvalga
Viena iš efektyviausių modelio architektūrų šnekamosios kalbos atpažinimo problemai spręsti giliuoju mokymusi yra Conformer'io architektūra. Ši architektūra kombinuoja globalius bei lokalius ryšius tarp garso įrašo sekos elementų, integruodama konvoliucinius sluoksnius transmofmerio struktūroje. Tokia architektūra buvo pademonstruota pasiekianti *state-of-the-art* lygio našumą šnekos atpažinimo sprendime, pralenkianti transformerio bei konvoliucinio neuroninio tinklo modelių našumą[1].

Garso įrašai yra laiko eilučių pavidalo duomenys, tačiau tokių duomenų palyginimas erdvės - laiko dimensijose yra sudėtinga užduotis. Vienas iš algoritmų sekos palyginimui yra Dynamic Time Warping (DTW), kuris buvo pademonstruotas kaip veikiantis algoritmas šnekamosios kalbos žodžių atpažinimui[2]. Tačiau kadangi modeliuose, kuriuose esantys svoriais turi būti optimizuojami, vienas iš paprasčiausiu optimizavimo algoritmų yra gradientinio nusileidimo algoritmas, efektyviam jo panaudojimui reikalinga apsiriboti diferencijuojamomis operacijomis modelyje. Standartinis DTW algoritmas nėra diferencijuojama operacija, tačiau egzistuoja DTW algoritmo diferencijuojamas pakaitalas - SoftDTW.

Klasifikuojant garso įrašus, svarbu yra savybių ištraukimas iš jo. Vienas iš žinomų savybių ištraukėjų garso duomenims yra *Mel Frequency Cepstrum Coefficients (MFCC)*, leidžiantis paversti *waveform* pavidalo duomenis į MFCC koeficientų seką. Alternatyvus būdas gali būti *wav2vec 2.0* modelis, atsižvelgiantis į visą garso įrašą ir ištraukiantis savybes vektorių sekos pavidalu. *Wav2vec 2.0* modelis, apmokytas savarankiškai-prižiūrimo mokymosi principu, gali pateikti garso segmento reprezentacijas, galimai labiau tinkamas tolimesniam apdorojimui, pvz. šnekamosios kalbos atpažinimui, pritaikant *Connectionist Temporal Classification* tinklą. Tokių reprezentacijų privalumas, lyginant su MFCC reprezentacijomis, yra tai, kad ištraukiant savybes su *wav2vec 2.0* modeliu, kiekvieno garso įrašo segmento vektorinė reprezentacija yra atsižvelgta pagal gretima segmento kontekstą garso įraše. MFCC metodas ištraukia segmento savybių vektorių neatsižvelgdamas į gretimus segmentus. [3]

#### Duomenų aibė
Žemiau pateiktas klasių skirstinys duomenų rinkiniuose, naudojamuose apmokymui, validavimui, testavimui.
![Image](https://i.imgur.com/SumaVn1.png)

Žemiau pateiktos pavyzdinės MFCC spektrogramos kiekvienai klasei.
![Image](https://i.imgur.com/pqtNPa0.png)

Iš pateiktų spektrogramų, paaiškinamų skirtumų tarp spektrogramų nesimato.

#### Modelio architektūra
Šiame darbe kurtas klasifikavimo modelis nenaudoja giliojo mokymosi savybių ištraukimui. Vietoje to, modelis naudoja prototipų mokymusi (angl. prototype learning) strategiją, mėgindamas rasti kiekvieną klasę idealiai reprezentuojantį atskaitos MFCC vektorių, vadinamą prototipu.

Signalo pavidalo (angl. waveform) garso įrašai buvo apdoroti prieš modelio apmokymą. Įrašai resample'inti į 16 kHz, sumažintas triukšmas, išlygintas garsumas, pritaikyta pre-emfazė (angl. pre-emphasis), suvienodinta trukmė iki 1 sek., transformuoti į MFCC koeficientų seką (13 koeficientų per segmentą), standartizuoti pagal kiekvieną MFCC koeficientą individualiai, apskaičiuota pirmos ir antros eilės MFCC koeficientų išvestinė. Bendrai gauta 3*13=39 koeficientai per laiko segmentą, kurių iš viso yra 81 per garso įrašą.

Klasifikuojant garso įrašą, matuojami atstumai tarp garso įrašo ir kiekvieną klasę reprezentuojančio prototipo, naudojant Dynamic Time Warping (DTW) kaip atstumo metriką, o klasė nustatoma pagal prototipą su žemiausia atstumo reikšme. Taigi, modelio parametrai yra 8 prototipų vektoriai.

Kadangi optimizuojant modelio parametrus naudojamas gradientinio nusileidimo metodas, o jam reikalingas gradientas, buvo pritaikytas diferencijuojamas DTW algoritmo variantas SoftDTW.

Buvo pritaikyta nuosava sugalvota neatitikties funkcija (angl. loss function)


$$
\text{L(x̂, y, W)} = \begin{cases} 
  \frac{D(x̂, p_y)}{m_x̂} & \text{jei } \text{argmin}(\{D(x̂, p_j) \,|\, j \in \{1, \ldots, M\}\}) \neq y \text{ (jei klasifikuota neteisingai)}  \\
  0 & \text{kitu atveju}
\end{cases}
$$

kur:
- \( x̂ \) yra duomenų taškas po išankstinio apdorojimo (angl. pre-processing)
- \( y \) yra tikrosios x̂ duomenų taško klasės etiketė
- \( W \) yra modelio svoriai
- \( D(i, j) \) yra Soft-DTW atstumas tarp \( i \)-tojo duomenų taško ir \( j \)-tojo prototipo
- \( M \) yra prototipų (klasių) kiekis
- \( p_y \) yra prototipas, reprezentuojantis teisingą duomenų taško klasės etiketę
- \( m_x̂ = \text{softmin} \{D(x̂, p_j) \,|\, j \in \{1, \ldots, M\} \text{ and } j \neq y \} \) yra minimalus atstumas tarp x̂ ir netinkamos klasės prototipų

Dėl nenustatytų priežasčių, duomenų augmentavimas tikslumo apmokymo ar validavimo duomenų rinkiniuose nepadidino, o priešingai, reikšmingai sumažino, dėl to duomenų augmentavimas nebuvo pritaikytas.

#### Apmokymo procesas
Apmokymas buvo atliktas naudojant skaičiavimo resursus iš Macbook Pro 2018 i7 (be GPU) kompiuterio, MacOS operacinės sistemos aplinkoje. Modelio sudarymui buvo naudojama `Python3.11`, `PyTorch`, `torchaudio`, `pysdtw`. Buvo naudojami tokie hiperparametrai: mokymosi tempas (angl. learning rate) `0.1` (krentantis žemyn), SoftDTW γ (gamma) parametras `0.1`, MFCC koeficientų kiekis garso įrašo segmentui `13`. Apmokymo trukmė 1.5h

#### Rezultatų analizė
Po 33 epochų, klasifikavimo tikslumas apmokymo duomenų rinkinyje buvo 98.6%, tikslumas validavimo duomenų rinkinyje 95%, tikslumas testavimo duomenų rinkinyje 95%.

Testavimo rezultatai su šiame darbe sukurtu klasifikavimo modeliu:

![Image](https://i.imgur.com/gJZCx8A.png)

Lyginant šiame darbe padarytą modelį su pavyzdiniu konvoliuciniu neuroniniu tinklu paremtu modeliu

Testavimo rezultatai su pavyzdiniu CNN modeliu:
![Image](https://www.tensorflow.org/static/tutorials/audio/simple_audio_files/output_LvoSAOiXU3lL_0.png)

#### Išvados
Nenaudojant gilaus mokymosi, buvo sėkmingai sukurtas garso įrašų klasifikavimo modelis, pritaikant Dynamic Time Warping (DTW) algoritmo SoftDTW modifikaciją. Darbas parodo, kad gilusis mokymas nėra būtinas, norint klasifikuoti naudotino duomenų rinkinio garso įrašus su anksčiau pateiktu tikslumu. Sprendimą būtų įmanoma patobulinti leidžiant kiekvienai klasei turėti kelis prototipus, neabsiribojant vienu.

#### Literatūros šaltiniai
[1] Conformer: Convolution-augmented Transformer for Speech Recognition (16 May 2020), Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, Ruoming Pang
[2] Speech recognition using Dynamic Time Warping (DTW) (2019), Yurika Permanasari, Erwin H. Harahap and Erwin Prayoga Ali
[3] wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (2020), Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli
[4] Simple audio recognition: Recognizing keywords, https://www.tensorflow.org/tutorials/audio/simple_audio
[5] Robust Classification with Convolutional Prototype Learning (2018), Hong-Ming Yang, Xu-Yao Zhang, Fei Yin, Cheng-Lin Liu
# AMES Stereo Basic Pipeline

## 1. Преобразование в формат ISIS

```shell
mroctx2isis from = P03_002258_1817_XI_01N356W.IMG to = P03_002258_1817.cub
```

## 2. Инициализация метаданных
Для корректной работы команд необходимо предзагрузить IsisData отсюда: https://astrogeology.usgs.gov/docs/how-to-guides/environment-setup-and-maintenance/isis-data-area/

Т.к. работаем с миссей Mars Reconnaissance Orbiter, нужно ввести

```shell
downloadIsisData mro $ISISDATA
```

Также необходимо загрузить base данные

```shell
downloadIsisData base $ISISDATA
```

Инициализация:

```shell
spiceinit from = P03_002258_1817.cub web = true
```

## 3. Калибровка

```shell
ctxcal from = P03_002258_1817.cub to = P03_002258_1817.cal.cub
```

## 4. cam2map4stereo

```shell
cam2map4stereo.py P02_001981_1823.cal.cub P03_002258_1817.cal.cub
```

## 5. Основной этап - запуск пайплайна

```shell
parallel_stereo --stereo-algorithm asp_mgm \
 --subpixel-mode 9 \
 --corr-memory-limit-mb 16384 \
 --processes 4 \
 --threads-multiprocess 8 \
 --entry-point 1 \
 --stop-point 2 \
 P02_001981_1823.map.cub P03_002258_1817.map.cub results/out
```

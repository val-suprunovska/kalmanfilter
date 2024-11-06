# Дослідження фільтра Калмана
## Матриця коваріації шуму процесу (Q):
Малі значення Q (наприклад, 0.1, 1): фільтр вважає, що процес стабільний, і не допускає значних змін стану, тобто він більше покладається на вимірювання.
<img src="screenshots/01.png">
<img src="screenshots/02.png">

Збільшення Q (наприклад, 10, 100): фільтр стає більш чутливим до змін у процесі, дозволяючи більшу варіативність передбачень. Це може бути корисним для динамічних сигналів, але призводить до менш ефективного пригнічення шуму.
<img src="screenshots/04.png">
<img src="screenshots/03.png">

Отже, дисперсія шуму після фільтрації зросте при збільшенні Q, оскільки фільтр почне більше довіряти своєму передбаченню, а не вимірюванням.
## Матриця коваріації шуму вимірювання (R):
Малі значення R (наприклад, 1): фільтр більше довіряє вимірюванням і менш схильний до передбачень.
<img src="screenshots/05.png">

Великі значення R (наприклад, 50, 100): фільтр менш довіряє вимірюванням і більше покладається на свої передбачення.
<img src="screenshots/06.png">
<img src="screenshots/07.png">

## Початкова матриця коваріації (P):
Малі значення P (0.1): фільтр швидко починає довіряти своїм оцінкам, але може бути занадто впевненим.
<img src="screenshots/08.png">

Великі значення P (10, 100): фільтр потребує більше часу для адаптації, але може бути менш схильним до надмірної корекції.
<img src="screenshots/09.png">
<img src="screenshots/10.png">

## Початкова оцінка стану (Initial state estimate):
Суттєва різниця між початковою оцінкою та справжнім сигналом може призводити до повільної збіжності фільтра, але з часом фільтр все одно стабілізується.
<img src="screenshots/11.png">
<img src="screenshots/12.png">

Точне початкове значення дозволяє фільтру швидко вийти на правильну траєкторію оцінок.

## Постійна складова сигналу (offset):
Зміна постійної складової дозволяє оцінити, чи добре фільтр справляється з адаптацією до зміщених сигналів.  
offset = 5
<img src="screenshots/13.png">
offset = 20
<img src="screenshots/14.png">

Великі значення зсуву можуть призвести до тимчасової розбіжності, але за належної конфігурації фільтр має коригувати оцінки відповідно до нового зміщення.
## Загальний час моделювання (total_time):
Збільшення загального часу моделювання дозволяє оцінити, як фільтр стабілізується за триваліших періодів і чи достатньо часу для адаптації.  
total_time = 0.5
<img src="screenshots/15.png">
total_time = 2
<img src="screenshots/16.png">

Короткий час моделювання може призвести до недостатньої адаптації фільтра.

## Дослідження частоти сигналу (frequency)
Зміна частоти сигналу впливає на те, наскільки швидко змінюється сигнал. Низька частота означає повільні коливання, тоді як висока частота призводить до швидких коливань.  
frequency = 5
<img src="screenshots/17.png">
## Дослідження амплітуди сигналу (amplitude)
Зміна амплітуди сигналу впливає на його інтенсивність або максимальне відхилення від середнього значення.  
При високій амплітуді фільтр може адаптуватися повільніше, особливо якщо параметри фільтра (наприклад, матриця P або R) не налаштовані під такі коливання.  
amplitude = 20
<img src="screenshots/18.png">
## Дослідження інтервалу дискретизації (sampling_interval)
Інтервал дискретизації визначає, як часто знімаються вимірювання. Менший інтервал означає частіші вимірювання, а більший інтервал – рідші.  
sampling_interval = 0.0005
<img src="screenshots/19.png">

Занадто часті вимірювання можуть призвести до високої частоти оновлення, що може бути добре для точності, але потребує більше обчислювальних ресурсів.  
sampling_interval = 0.1
<img src="screenshots/20.png">

Занадто великий інтервал призводить до того, що фільтр може не встигати реагувати на зміни сигналу і давати запізнілі оцінки.
## Дослідження матриці переходу стану (F) та матриці вимірювання (H)
Матриця F визначає, як поточний стан залежить від попереднього. Наприклад, для простих випадків вона може бути одиничною, але для більш складних моделей може включати інші коефіцієнти.

Занадто низьке значення призводить до надмірної консервативності фільтра, він буде повільно адаптуватися до змін.  
F = 0.5
<img src="screenshots/21.png">

Високе значення може зробити фільтр нестабільним, він може надто реагувати на зміни сигналу.  
F = 1.5
<img src="screenshots/22.png">

---
Матриця H визначає, як вимірювання відповідають стану системи. Це допомагає фільтру адаптуватися до виміряних значень.

Значення менше 1 призводять до меншої довіри до вимірювань, що змушує фільтр більше покладатися на свої передбачення.
<img src="screenshots/23.png">

Значення більше 1 підвищують чутливість фільтра до вимірювань, що може призвести до швидкої адаптації, але й до вищої нестабільності за наявності шуму.
<img src="screenshots/24.png">

## Висновок

Фільтр Калмана досягає меншої дисперсії при збалансованих значеннях параметрів Q і R та при оптимальній початковій матриці P. Зменшення Q і R за умови високої якості моделі та вимірювань сприяє більш стабільним та точним оцінкам.  
Параметри:  
+ Q = 1
+ R = 1000
+ P = 1000
+ x = 9
<img src="screenshots/25.png">
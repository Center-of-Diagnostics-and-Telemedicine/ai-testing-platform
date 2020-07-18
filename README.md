# Medical AI Testing Platform
A platform allows the developers of AI-based solutions for medicine to perform analytical validation of their algorithms on a reference dataset in an automatic way.

## Challenges
Before deployment of an AI-based software (called _AI service_) in medical practice, it is crucial to determine whether the technology is capable and fit-for-purpose for the intended use. Such measures are called analytical validation. Only services that pass the analytical validation should be allowed to further access to real-world medical systems.

The integration is a challenging and costly process, since medical equipment is a demanding domain, and the proper setup requires competent experts. To sum up, the integration cannot be initiated, unless an AI service has been thoroughly tested.

Currently, the analytical validation is either semi-automated or performed manually. Lack of automation allows random or intentional deviations from the analitycal validation scenario and creates opportunities for possible manipulations of the reference data. This all results in unequal conditions for the testers and, thus, biased results.

## Solution
We suggest a simple web service for orchestrating and monitoring the analytical validation. 

The main features are:
* token-based authorisation;
* unified rules of validation process;
* unlimited access to the data set for an algorithm adjustment;
* registration of the number of the test attempts;
* registration of the time spent on the processing of a single element of the reference database;
* standardisation of metrics, building ROC-curves and performing the real-time assessment.

## Database Schema
```SQL
CREATE TABLE services (id INTEGER NOT NULL PRIMARY KEY, name VARCHAR(100), token VARCHAR(64));
CREATE TABLE session_tokens (id INTEGER NOT NULL PRIMARY KEY, service INTEGER, session_token VARCHAR(64), issue_date TIMESTAMP, expiry_date TIMESTAMP, active BOOLEAN);
CREATE TABLE testing (id INTEGER NOT NULL PRIMARY KEY, session INTEGER, dataset_title TEXT, dataset_file_id INTEGER, created TIMESTAMP, retrieved TIMESTAMP, received TIMESTAMP, ai_ct INTEGER, ai_left_affected_part FLOAT, ai_left_total_volume FLOAT, ai_left_affected_volume FLOAT, ai_right_affected_part FLOAT, ai_right_total_volume FLOAT, ai_right_affected_volume FLOAT, viewer_url VARCHAR(200), description VARCHAR(300), requests INTEGER);
CREATE TABLE datasets(id INTEGER NOT NULL PRIMARY KEY, title TEXT, filename TEXT, var1 VARCHAR(30), var2 VARCHAR(30), var3 VARCHAR(30), var4 VARCHAR(30), var5 VARCHAR(30), added TIMESTAMP);
```

## Credits
* Nikolay Pavlov<sup name="a1">[1](#f1)</sup>, MD, MPA (n.pavlov@npcmr.ru);
* Anna Andreychenko<sup name="a1">[1](#f1)</sup>, PhD (a.andreychenko@npcmr.ru);
* Sergey Morozov<sup name="a1">[1](#f1)</sup>, MD, PhD, MPH, CIIP (morozov@npcmr.ru).

<span id="f1">\[1\]</span> – [Research and Practical Clinical Center of Diagnostics and Telemedicine Technologies](https://tele-med.ai), Department of Health Care of Moscow, Russia

## Version
0.1 (alpha).

## License
This code is licensed under the [Apache License 2.0](LICENSE).

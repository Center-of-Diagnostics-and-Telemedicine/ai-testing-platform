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

## Credits
* Nikolay Pavlov, CDO, Research and Practical Clinical Center of Diagnostics and Telemedicine Technologies, Department of Health Care of Moscow (n.pavlov@npcmr.ru).
* Anna Andreychenko, Head of Sector for Medical Informatics, Radiomics and Radiogenomics, Research and Practical Clinical Center of Diagnostics and Telemedicine Technologies, Department of Health Care of Moscow (a.andreychenko@npcmr.ru).
* Sergey Morozov, CEO, Radiomics and Radiogenomics, Research and Practical Clinical Center of Diagnostics and Telemedicine Technologies, Department of Health Care of Moscow (a.andreychenko@npcmr.ru).

1 – 

## Version
0.1 (alpha).

## License
This code is licensed under the [Apache License 2.0](LICENSE).

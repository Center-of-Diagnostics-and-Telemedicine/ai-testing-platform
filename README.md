# AI Testing Platform
The platform allows the developers of AI-based to perform analytical validation of their algorithms in automatic way.

## Problem
Before integrating an AI-based software (called _service_) in applied medicine, it is crucial to determine whether the technology is capable and fit-for-purpose for the intended use. Such measures are called analytical validation. Only the services that pass the analytical validation should be allowed to further access to real-world medical systems.

The integration is a challenging and costly process, since medical equipment is a demanding domain, and the proper setup requires competent experts. To sum up, the integration cannot be initiated, unless the AI-based service has been thoroughly tested.

Currently, the analytical validation is either semi-automated or performed manually. Lack of automation allows random or intentional departures from the test scenario, provides comprehensive control over the manipulation of the reference data and creates an uneven playing field for the testers.

## Solution
We suggest the simple web service for orchestrating the analytical validation. 

The main features are:
* token-based authorisation;
* unified rules of validation process;
* unlimited access to the data set for algorithm adjustment;
* registration of the number of test attempts;
* registration of the time spent on the processing of a single data element;
* standartisation of metrics, building ROC-curves and performing the real-time assessment.

## Credits
Nikolay Pavlov, CDO, Research and Practical Clinical Center of Diagnostics and Telemedicine Technologies, Department of Health Care of Moscow (n.pavlov@npcmr.ru).

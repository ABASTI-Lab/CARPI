# CARPI Automated Framework
The **Cancer Radiomic and Perfusion Imaging (CARPI)** automated framework is a Python-based software application that is vendor- and sequence-neutral. CARPI uses DICOM-RT files generated using an application of the user's choice and automatically performs: <br />
**&nbsp; :heavy_check_mark: Radiomic Feature Extraction <br />
&nbsp; :heavy_check_mark: Perfusion Analysis <br />
&nbsp; :heavy_check_mark: Relational Database Generation <br />
&nbsp; :heavy_check_mark: Graphic Report Generation** <br />

> [!NOTE]
> **Link to Technical Paper: https://ascopubs.org/doi/10.1200/CCI.23.00118**

![carpi_reports](https://github.com/ABASTI-Lab/CARPI/assets/143734103/78a2bfd5-3665-4ac5-adc7-0917687cbac9)

# Steps to Successfully Run CARPI:
**STEP 1.** Download the CARPI directory from this repository. <br />
**STEP 2.** Install the required Python libraries: <br />
&emsp;&emsp; a) pip install pydicom <br />
&emsp;&emsp; b) pip install DicomRTTool <br />
&emsp;&emsp; c) pip install pyradiomics <br />
&emsp;&emsp; d) pip install psycopg2 <br />
**STEP 3.** Download and configure PostgreSQL: https://www.postgresql.org/download/ <br />
**STEP 4.** Open the **database.ini** file and edit it according to your PostgreSQL configuration: <br />

<img width="243" alt="database_file" src="https://github.com/ABASTI-Lab/CARPI/assets/143734103/8c9ef795-e371-4036-9d82-4f8ff63c450f">

**STEP 5.** Open the **CARPI_Main.py** script and enter the user-defined variables: <br />

<img width="540" alt="carpi_main" src="https://github.com/ABASTI-Lab/CARPI/assets/143734103/20c0cd5a-ba2b-40d2-be5d-1482e0aae2d7">

**STEP 6.** Run **CARPI_Main.py**

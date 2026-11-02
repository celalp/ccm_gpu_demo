# CCM HPC GPU demonstration

This repository contains all the code and the enviromental setup requrired to replicate the results of the CCM 
"Accelerating research with gpus on high perfomance computing clusters" lecture. 

first you will need to install conda, you can do that by following the instrctions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

After that you can create each env like so:

```bash
conda env create -f environment/<env_name>.yaml
```

To replicate the results you will need to set up several conda environments because they have conflicting dependencies. 
The code in the "live demo" was done in the corresponding conda environments described in the "environment" folder. There
are a couple of redundancies within the environments because of trial and error. All the code that use the respective 
environments is in the "ccm_demo" folder.

If you have any questions or issues please crate an issues in this repository. 

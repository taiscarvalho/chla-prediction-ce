# Uncovering the influence of hydrological and climate variables in chlorophyll-a concentration in tropical reservoirs with machine learning
The codes contained here implement the methods of the paper "Uncovering the influence of hydrological and climate variables in chlorophyll-a concentration in tropical reservoirs with machine learning", submitted to Water Resources Research. The code includes the R implementation of the evaluated Machine Learning models, and the figures presented in the manuscript. Also, all data used in the paper is available in the file "dataset_chla_ceara.csv".

## Requirements
Before running the code, you need to install the following packages:
```
install.packages(caret)
install.packages(dplyr)
install.packages(magrittr)
install.packages(randomForest)
install.packages(corrplot)
install.packages(ggplot2)
install.packages(pdp)
install.packages(ggsci)
install.packages(ggradar)
```

## Graphical abstract

![alt text](https://raw.github.com/taiscarvalho/ml_waterdemand/main/graphical-abstract.png)


## License
These codes are free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published bythe Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.

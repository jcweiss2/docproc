library(cluster)
library(dplyr)

clusters_tb = read.csv("180415clusters.csv") %>% tbl_df

# num cols:
# $ cat p6iedds3xiaht1nmhlval5tzq0k9dqcx|head -n 2|tail -n 1|sed 's/[^,]//g'| wc -c
# 1582
classes = c("numeric", rep("numeric", 1581))
tb = read.csv("p6iedds3xiaht1nmhlval5tzq0k9dqcx", colClasses=classes)
tb = tb %>% tbl_df
saveRDS(tb, file="p6iedds3xiaht1nmhlval5tzq0k9dqcx.rds")

# Make matrix
mat = as.matrix(tb[,2:ncol(tb)])
rownames(mat) = as.character(tb$npi)

# Make distance
D = dist(mat)

library(cluster)
library(dplyr)

# first run step 1, then:

pdf("180415clusters.pdf")

clusters = as.integer(clusters_tb$cluster)
si = silhouette(clusters)
plot(si)

dev.off()

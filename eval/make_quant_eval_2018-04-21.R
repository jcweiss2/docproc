library(cluster)
library(clusterCrit)
library(dplyr)
library(proxy)

# Set options.
#dist_tag = "cosine"
dist_tag = "Euclidean"
#space_tag = "hidden"
space_tag = "output"
tag = sprintf("space_%s.dist_%s", space_tag, dist_tag)

# Determine output locations.
silhouette_filename = sprintf("silhouette.%s.pdf", tag)
ext_tb_filename = sprintf("external_criteria_on_subset.%s.csv", tag)
int_tb_filename = sprintf("internal_criteria_on_subset.%s.csv", tag)
best_int_criteria_filename = sprintf("best_int_criteria_on_subset.%s.csv", tag)

# Load clusters.
if (!("clusters_tb" %in% ls())) {
  cat("Loading clusters_tb\n")
  clusters_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/180418clustersGood.csv.gz"), stringsAsFactors=FALSE) %>% tbl_df
  clusters_tb = clusters_tb %>% select(-X) %>% mutate(cluster = as.integer(cluster))
}

# Load kmeans clusters.
if (!("kmeans_tb" %in% ls())) {
  cat("Loading kmeans_tb\n")
  kmeans_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/kmeans_2018-04-20.csv.gz"), stringsAsFactors=FALSE) %>% tbl_df
  kmeans_tb = kmeans_tb %>% select(-X) %>% mutate(kmeans_cluster = as.integer(cluster)) %>% select(-cluster)
}

# Load providers data.
if (!("providers_tb" %in% ls())) {
  cat("Loading providers_tb\n")
  providers_tb = readRDS("../data/Medicare_Provider_Utilization_and_Payment_Data__Part_D_Prescriber_Summary_Table_CY2015.rds")
}

# Load hidden data.
if (!("hidden_tb" %in% ls())) {
  cat("Loading hidden_tb\n")
  col_classes = c(rep("numeric", 128), "integer")
  names(col_classes) = c(sprintf("X%d", 0:127), "npi")
  hidden_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/180418npihiddenGood.csv.gz"), colClasses=col_classes) %>% tbl_df
  hidden_x = as.matrix(hidden_tb[,sprintf("X%d", 0:127)])
  rownames(hidden_x) = hidden_tb$npi
}

# Load output data.
if (!("output_tb" %in% ls())) {
  cat("Loading output_tb\n")
  header_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/p6iedds3xiaht1nmhlval5tzq0k9dqcx.gz | head -n 2")) %>% tbl_df
  output_col_classes = c("integer", rep("numeric", ncol(header_tb)-1))
  output_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/p6iedds3xiaht1nmhlval5tzq0k9dqcx.gz"), colClasses=output_col_classes) %>% tbl_df
  output_x = as.matrix(output_tb %>% select(-npi))
  rownames(output_x) = output_tb$npi
}

specialties_tb = providers_tb %>% select(npi, specialty_description) %>%
                                  mutate(specialty_id = as.integer(specialty_description))


both_tb = clusters_tb %>% full_join(kmeans_tb, by="npi") %>%
                          full_join(specialties_tb, by="npi")

# Subsample 1% of the providers.
#sub_npi = unlist(sapply(unique(clusters_tb$cluster), function(c) {
#  rn = (clusters_tb %>% filter(cluster == c))$npi
#  sub_rn = sample(rn, round(length(rn)*0.01))
#  sub_rn
#}))
sub_npi = sample(clusters_tb$npi, round(length(clusters_tb$npi)*0.01))
sub_both_tb = both_tb %>% filter(npi %in% sub_npi)
sub_hidden_x = hidden_x[as.character(sub_npi),]

# Compute distance.
if (space_tag == "hidden") {
  sub_dist = dist(sub_hidden_x, sub_hidden_x, method=dist_tag)
} else if (space_tag == "output") {
  sub_dist = dist(sub_hidden_x, sub_hidden_x, method=dist_tag)
} else {
  stop(sprintf("Unknown space_tag: %s", space_tag))
}

# Make silhoutte plot.
cluster_sil = silhouette(sub_both_tb$cluster, sub_dist)
kmeans_sil = silhouette(sub_both_tb$kmeans_cluster, sub_dist)
speciality_sil = silhouette(sub_both_tb$specialty_id, sub_dist)
pdf(silhouette_filename)
plot(cluster_sil, main="Silhouette Plot for Our Clustering")
plot(kmeans_sil, main="Silhouette Plot for K-Means")
plot(speciality_sil, main="Silhouette Plot based on Specialties")
dev.off()

# Carry out external comparisons.
ext_tb1 = extCriteria(sub_both_tb$cluster, sub_both_tb$specialty_id, "all") %>% as_data_frame %>% mutate(comparison="cluster_to_specialty")
ext_tb2 = extCriteria(sub_both_tb$kmeans_cluster, sub_both_tb$specialty_id, "all") %>% as_data_frame %>% mutate(comparison="kmeans_to_specialty")
ext_tb3 = extCriteria(sub_both_tb$cluster, sub_both_tb$kmeans_cluster, "all") %>% as_data_frame %>% mutate(comparison="cluster_to_kmeans")
ext_tb = bind_rows(ext_tb1, ext_tb2, ext_tb3)
ext_tb = ext_tb %>% select(comparison, everything())
write.csv(ext_tb, file=ext_tb_filename, row.names=FALSE)

# Carry out internal comparisons.
int_tb1 = intCriteria(sub_hidden_x, sub_both_tb$cluster, "all") %>% as_data_frame %>% mutate(method="cluster")
int_tb2 = intCriteria(sub_hidden_x, sub_both_tb$kmeans_cluster, "all") %>% as_data_frame %>% mutate(method="kmeans")
int_tb3 = intCriteria(sub_hidden_x, sub_both_tb$specialty_id, "all") %>% as_data_frame %>% mutate(method="specialty")
int_tb = bind_rows(int_tb1, int_tb2, int_tb3)
int_tb = int_tb %>% select(method, everything())
write.csv(int_tb, file=int_tb_filename, row.names=FALSE)

# Determine which method is best for each internal criterion.
best_int_criteria = sapply(colnames(int_tb %>% select(-method)), function(crit) {
  x = int_tb[[crit]]
  idx = bestCriterion(x, crit)
  if (is.nan(idx)) {
    label = NA
  } else {
    label = int_tb$method[[idx]]
  }
  label
})
best_int_criteria = data_frame(criterion=names(best_int_criteria), best_method=best_int_criteria)
write.csv(best_int_criteria, file=best_int_criteria_filename, row.names=FALSE)

# Tabulate best.
print(best_int_criteria %>% group_by(best_method) %>% summarise(n = n()))

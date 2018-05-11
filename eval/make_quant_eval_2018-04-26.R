library(cluster)
library(clusterCrit)
library(dplyr)
library(flashpcaR)
library(ggplot2)
library(proxy)
library(tsne)

# Set options.
dist_tag = "cosine"
#dist_tag = "Euclidean"
space_tag = "hidden"
#space_tag = "output"
date_tag = "2018-04-26"
#todo_tag = "int"
#todo_tag = "ext"
#todo_tag = "silhouette"
todo_tag = "tsne"
#todo_tag = "flashpca"

# Determine output locations.
silhouette_filename = sprintf("../results/silhouette.space_%s.dist_%s.date_%s.pdf", space_tag, dist_tag, date_tag)
tsne_filename = sprintf("../results/tsne.space_%s.dist_%s.date_%s.pdf", space_tag, dist_tag, date_tag)
tsne_result_filename = sprintf("../results/tsne_result.space_%s.dist_%s.date_%s.pdf", space_tag, dist_tag, date_tag)
flashpca_filename = sprintf("../results/flashpca.space_%s.dist_%s.date_%s.pdf", space_tag, dist_tag, date_tag)
ext_tb_filename = sprintf("../results/external_criteria_on_subset.date_%s.csv", date_tag)
int_tb_filename = sprintf("../results/internal_criteria_on_subset.space_%s.date_%s.csv", space_tag, date_tag)
#best_int_criteria_filename = sprintf("best_int_criteria_on_subset.%s.csv", tag)

# Load clusters.
if (!("clusters_tb" %in% ls())) {
  cat("Loading clusters_tb\n")
  clusters_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/180426clusters60merged.csv.gz"), stringsAsFactors=FALSE) %>% tbl_df
  clusters_tb = clusters_tb %>% select(-X) %>% mutate(cluster = as.integer(cluster))
}

# Load kmeans clusters.
if (!("kmeans_tb" %in% ls())) {
  cat("Loading kmeans_tb\n")
  kmeans_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/clusters_kmeans_procs_2018-04-26.gz"), stringsAsFactors=FALSE) %>% tbl_df
  kmeans_tb = kmeans_tb %>% select(-X) %>% mutate(kmeans_cluster = as.integer(cluster)) %>% select(-cluster)
}

# Load providers data and make specialties data.
if (!("specialties_tb" %in% ls())) {
  cat("Loading providers_tb and making specialties_tb.\n")
  providers_tb = readRDS("../data/Medicare_Provider_Utilization_and_Payment_Data__Part_D_Prescriber_Summary_Table_CY2015.rds")
  specialties_tb = providers_tb %>% select(npi, specialty_description) %>%
                                    mutate(specialty_id = as.integer(specialty_description))
}

# Make both_tb.
cat("Make both_tb.\n")
both_tb = clusters_tb %>% full_join(kmeans_tb, by="npi") %>%
                          full_join(specialties_tb, by="npi")

# Load space (covariate) data if needed.
if (todo_tag %in% c("silhouette", "tsne", "flashpca", "int")) {

  # Load hidden data if needed.
  if (space_tag == "hidden") {
    if (!("hidden_tb" %in% ls())) {
      cat("Loading hidden_tb\n")
      hidden_col_classes = c(rep("numeric", 128), "integer")
      names(hidden_col_classes) = c(sprintf("X%d", 0:127), "npi")
      hidden_tb = read.csv(pipe("tar -xOzf ../data/180426npihidden60proc.tar.gz 180426npihidden60proc.csv"), colClasses=hidden_col_classes) %>% tbl_df
      hidden_x = as.matrix(hidden_tb[,sprintf("X%d", 0:127)])
      rownames(hidden_x) = hidden_tb$npi
    }
    space_x = hidden_x

  # Load output data if needed.
  } else if ((space_tag == "output") & (todo_tag %in% c("silhouette", "int"))) {
    if (!("output_tb" %in% ls())) {
      cat("Loading output_tb\n")
      header_tb = read.csv(pipe("tar -xOzf ../data/docprocwide.csv.tar.gz medicare/docprocwide.csv | head -n 2")) %>% tbl_df
      output_col_classes = c("integer", rep("numeric", ncol(header_tb)-1))
      output_tb = read.csv(pipe("tar -xOzf ../data/docprocwide.csv.tar.gz medicare/docprocwide.csv"), colClasses=output_col_classes) %>% tbl_df
      output_x = as.matrix(output_tb %>% select(-npi))
      rownames(output_x) = output_tb$npi
    }
    space_x = output_x

  } else {
    stop(sprintf("Unknown space_tag: %s", space_tag))
  }

  # Subsample 1% of the providers.
  cat("Subsample 1% of the providers.\n")
  #sub_npi = unlist(sapply(unique(clusters_tb$cluster), function(c) {
  #  rn = (clusters_tb %>% filter(cluster == c))$npi
  #  sub_rn = sample(rn, round(length(rn)*0.01))
  #  sub_rn
  #}))
  sub_npi = sample(clusters_tb$npi, round(length(clusters_tb$npi)*0.01))
  #sub_npi = sample(clusters_tb$npi, round(length(clusters_tb$npi)*0.10))
  sub_both_tb = both_tb %>% filter(npi %in% sub_npi)
  sub_space_x = space_x[as.character(sub_npi),]
}

# Compute distance.
if (todo_tag %in% c("silhouette", "tsne")) {
  cat("Compute distance.\n")
  sub_dist = dist(sub_space_x, sub_space_x, method=dist_tag)
}

# Make silhoutte plot.
if (todo_tag == "silhouette") {
  cat("Make silhoutte plot.\n")
  cluster_sil = silhouette(sub_both_tb$cluster, sub_dist)
  kmeans_sil = silhouette(sub_both_tb$kmeans_cluster, sub_dist)
  speciality_sil = silhouette(sub_both_tb$specialty_id, sub_dist)
  pdf(silhouette_filename)
  plot(cluster_sil, main="Silhouette Plot for Our Clustering")
  plot(kmeans_sil, main="Silhouette Plot for K-Means")
  plot(speciality_sil, main="Silhouette Plot based on Specialties")
  dev.off()
}

# Make tsne plot.
if (todo_tag == "tsne") {
  cat("Make tsne plot.\n")
  sub_tsne_result = tsne(sub_dist)
  sub_tsne_tb = sub_tsne_result %>% `colnames<-`(c("tsne_dim1", "tsne_dim2")) %>% as_data_frame
  stopifnot(nrow(sub_tsne_tb) == nrow(sub_both_tb))
  sub_tsne_both_tb = bind_cols(sub_tsne_tb, sub_both_tb)
  saveRDS(list(sub_tsne_result = sub_tsne_result, 
               sub_npi = sub_npi,
               sub_tsne_both_tb = sub_tsne_both_tb), file=tsne_result_filename)
  pdf(tsne_filename)
  print(ggplot(sub_tsne_both_tb, aes(x = tsne_dim1, y = tsne_dim2, label = as.character(cluster))) + geom_text(check_overlap=TRUE) + ggtitle("t-SNE colored by our clustering"))
  print(ggplot(sub_tsne_both_tb, aes(x = tsne_dim1, y = tsne_dim2, label = as.character(kmeans_cluster))) + geom_text(check_overlap=TRUE) + ggtitle("t-SNE colored by k-means"))
  print(ggplot(sub_tsne_both_tb, aes(x = tsne_dim1, y = tsne_dim2, label = as.character(specialty_id))) + geom_text(check_overlap=TRUE) + ggtitle("t-SNE colored by specialty assignment"))
  dev.off()
}

# Make flashpca plot.
if (todo_tag == "flashpca") {
  cat("Make flashpca plot.\n")
  sub_pca_result = flashpca(sub_space_x, ndim=2, stand="sd")
  sub_pca_mat = sub_pca_result$projection
  sub_pca_tb = sub_pca_mat %>% `colnames<-`(c("pca_dim1", "pca_dim2")) %>% as_data_frame
  stopifnot(nrow(sub_pca_tb) == nrow(sub_both_tb))
  sub_pca_both_tb = bind_cols(sub_pca_tb, sub_both_tb)
  pdf(flashpca_filename)
  print(ggplot(sub_pca_both_tb, aes(x = pca_dim1, y = pca_dim2, color = factor(cluster))) + geom_point() + ggtitle("PCA colored by our clustering"))
  print(ggplot(sub_pca_both_tb, aes(x = pca_dim1, y = pca_dim2, color = factor(kmeans_cluster))) + geom_point() + ggtitle("PCA colored by k-means"))
  print(ggplot(sub_pca_both_tb, aes(x = pca_dim1, y = pca_dim2, color = factor(specialty_id))) + geom_point() + ggtitle("PCA colored by specialty assignment"))
  dev.off()
}

# Carry out external comparisons.
if (todo_tag == "ext") {
  cat("Carry out external comparisons.\n")
  ext_tb1 = extCriteria(both_tb$cluster,        both_tb$specialty_id, "all") %>% as_data_frame %>% mutate(comparison="cluster_to_specialty")
  ext_tb2 = extCriteria(both_tb$kmeans_cluster, both_tb$specialty_id, "all") %>% as_data_frame %>% mutate(comparison="kmeans_to_specialty")
  ext_tb3 = extCriteria(both_tb$cluster,        both_tb$kmeans_cluster, "all") %>% as_data_frame %>% mutate(comparison="cluster_to_kmeans")
  ext_tb = bind_rows(ext_tb1, ext_tb2, ext_tb3)
  ext_tb = ext_tb %>% select(comparison, everything())
  write.csv(ext_tb, file=ext_tb_filename, row.names=FALSE)
}

# Carry out internal comparisons.
if (todo_tag == "int") {
  cat("Carry out internal comparisons.\n")
  int_tb1 = intCriteria(sub_space_x, both_tb$cluster, "all") %>% as_data_frame %>% mutate(method="cluster")
  int_tb2 = intCriteria(sub_space_x, both_tb$kmeans_cluster, "all") %>% as_data_frame %>% mutate(method="kmeans")
  int_tb3 = intCriteria(sub_space_x, both_tb$specialty_id, "all") %>% as_data_frame %>% mutate(method="specialty")
  int_tb = bind_rows(int_tb1, int_tb2, int_tb3)
  int_tb = int_tb %>% select(method, everything())
  write.csv(int_tb, file=int_tb_filename, row.names=FALSE)
}

# # Determine which method is best for each internal criterion.
# cat("Determine which method is best for each internal criterion.\n")
# best_int_criteria = sapply(colnames(int_tb %>% select(-method)), function(crit) {
#   x = int_tb[[crit]]
#   idx = bestCriterion(x, crit)
#   if (is.nan(idx)) {
#     label = NA
#   } else {
#     label = int_tb$method[[idx]]
#   }
#   label
# })
# best_int_criteria = data_frame(criterion=names(best_int_criteria), best_method=best_int_criteria)
# write.csv(best_int_criteria, file=best_int_criteria_filename, row.names=FALSE)

# # Tabulate best.
# cat("Tabulate best.\n")
# print(best_int_criteria %>% group_by(best_method) %>% summarise(n = n()))

library(cluster)
library(clusterCrit)
library(dplyr)

date_tag = "2018-04-24"
directory_tag = sprintf("simulations_%s", date_tag)

merged_ext_tb_filename = sprintf("%s/ext_tb.simulation.date%s.csv", output_dir, date_tag)
merged_int_tb_filename = sprintf("%s/int_tb.simulation.date%s.csv", output_dir, date_tag)

input_options_tb = read.csv(text="centroids,samples,dims,sd,explode
25,1000,1000,0.01,10000
25,1000,100000,0.01,10000
25,1000,1000,0.1,10000
25,1000,1000,0.01,1000000
100,1000,1000,0.01,10000
25,100,1000,0.01,10000", colClasses=rep("character", 5))

baseline_tb = data_frame(centroids="25", samples="1000", dims="1000", sd="0.01", explode="10000")
input_options_tb$input_display_tag = sapply(1:nrow(input_options_tb), function(i) {
  r = input_options_tb[i,]
  t = ""
  for (n in (colnames(r))) {
    if (r[[n]] != baseline_tb[[n]]) {
      stopifnot(t == "")
      t = sprintf("%s=%s", n, r[[n]])
    }
  }
  if (t == "") {
    t = "baseline"
  }
  t
})

#dist_tags_tb = data_frame(dist_tag = c("cosine", "Euclidean"))
output_options_tb = expand.grid(dist_tag = c("cosine"),
                                space_tag = c("hidden", "output"))

ext_tb_list = NULL
int_tb_list = NULL

for (input_row_idx in 1:nrow(input_options_tb)) {
for (output_row_idx in 1:nrow(output_options_tb)) {

ir = input_options_tb[input_row_idx,]
input_tag = sprintf("centroids%s_samples%s_dims%s_sd%s_explode%s",
                    ir$centroids, ir$samples, ir$dims, ir$sd, ir$explode)

or = output_options_tb[output_row_idx,]
output_tag = sprintf("%s_space_%s.dist_%s.date_%s",
                     input_tag, or$space_tag, or$dist_tag, date_tag)

cat(sprintf("Loading for output_tag %s\n", output_tag))

# Determine step 1's output locations.
output_dir = sprintf("../results/%s", directory_tag)
ext_tb_filename = sprintf("%s/external_criteria.%s.csv", output_dir, output_tag)
int_tb_filename = sprintf("%s/internal_criteria.%s.csv", output_dir, output_tag)
best_int_criteria_filename = sprintf("%s/best_int_criteria.%s.csv", output_dir, output_tag)

if (!file.exists(ext_tb_filename)) {
  cat(sprintf("NOTE: %s doesn't exist\n", ext_tb_filename))
} else {
  this_ext_tb = read.csv(ext_tb_filename, stringsAsFactors=FALSE) %>% tbl_df
  for (n in names(ir)) {
    this_ext_tb[[n]] = ir[[n]]
  }
  for (n in names(or)) {
    this_ext_tb[[n]] = or[[n]]
  }
  ext_tb_list = c(ext_tb_list, list(this_ext_tb))
}

if (!file.exists(int_tb_filename)) {
  cat(sprintf("NOTE: %s doesn't exist\n", int_tb_filename))
} else {
  this_int_tb = read.csv(int_tb_filename, stringsAsFactors=FALSE) %>% tbl_df
  for (n in names(ir)) {
    this_int_tb[[n]] = ir[[n]]
  }
  for (n in names(or)) {
    this_int_tb[[n]] = or[[n]]
  }
  int_tb_list = c(int_tb_list, list(this_int_tb))
}

}
}

ext_tb = bind_rows(ext_tb_list)
int_tb = bind_rows(int_tb_list)
ext_tb_slice = ext_tb %>% filter(space_tag == "hidden", dist_tag == "cosine") %>% select(-space_tag, -dist_tag)

write.csv(ext_tb_slice, file=merged_ext_tb_filename, row.names=FALSE)
write.csv(int_tb, file=merged_int_tb_filename, row.names=FALSE)

ext_tb_slice %>% select(input_display_tag, clustering, precision, recall, rand, jaccard) %>% print(n=50)

for (v in c("precision", "recall", "rand", "jaccard")) {
  cat(sprintf("\n%s:\n", v))
  ext_tb_slice %>% select(input_display_tag, clustering, quo_name(v)) %>% dcast(... ~ clustering, value.var=v) %>% tbl_df %>% print(n=50)
}

# Make Latex table.

tmp_list = NULL
for (v in c("precision", "recall", "rand", "jaccard")) {
  tmp = ext_tb_slice %>% select(input_display_tag, clustering, quo_name(v)) %>% dcast(... ~ clustering, value.var=v) %>% tbl_df
  tmp = list(tmp)
  names(tmp) = v
  tmp_list = c(tmp_list, tmp)
}


print_row = function(rowname) {
  cat(rowname)
  score_names = c("rand", "jaccard", "recall", "precision")
  clustering_names = c("ours_merged", "kmeans", "hidden_kmeans")
  for (score_name in score_names) {
    for (clustering_name in clustering_names) {
      tmp = tmp_list[[score_name]]
      mat = as.matrix(tmp %>% select(hidden_kmeans, kmeans, ours_merged))
      rownames(mat) = tmp$input_display_tag
      val = mat[rowname,clustering_name]
      if (val == max(mat[rowname,clustering_names])) {
        cat(sprintf(" & \\textbf{%0.2f}", round(val, 2)))
      } else {
        cat(sprintf(" & %0.2f", round(val, 2)))
      }
    }
  }
  cat("\\\\ \n")
}
cat("\\begin{tabular}{l|ccc|ccc|ccc|ccc}\n")
cat("\\hline\n")
cat("& \\multicolumn{3}{c}{Rand} & \\multicolumn{3}{c}{Jaccard} & \\multicolumn{3}{c}{Recall} & \\multicolumn{3}{c}{Precision} \\\\ \n")
cat("Data")
for (i in 1:4) {
  cat(" & Ours & OKM & HKM")
}
cat("\\\\ \n")
cat("\\hline\n")
print_row("baseline")
print_row("samples=100")
print_row("dims=100000")
print_row("explode=1000000")
print_row("centroids=100")
print_row("sd=0.1")
cat("\\hline\n")
cat("\\end{tabular}\n")


fit <- function(X, y, output_dir, class_order=NULL, row_weights=NULL, ...){}

transform <- function(X, transformer, y=NULL, ...){
    # Ignore the model and convert the sparse input to dense
    as.data.frame(as.matrix(X))
}
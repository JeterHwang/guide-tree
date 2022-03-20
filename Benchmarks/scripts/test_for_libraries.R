libraries <- c("tidyverse","ggplot2","stringi","cowplot")
Sys.info()["nodename"]
for(library in libraries) 
{ 
  f = is.element(library, installed.packages()[,1])
  print(paste("Library",library, "is installed?", f))
  if(!f)
  {
    message("Missing library:",library )
    quit(status=1)
  }
}
quit(status=0)

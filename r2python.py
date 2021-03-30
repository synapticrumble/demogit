

rscript = """
library(ggplot2)
theme_set(theme_bw())

g <- ggplot(dataset, aes(class, cty))
g + geom_violin() + 
  labs(title="Violin plot", 
       subtitle="City Mileage vs Class of vehicle",
       caption="Source: mpg",
       x="Class of Vehicle",
       y="City Mileage")
"""

from pyensae.languages import r2python
print(r2python(rscript, pep8=True))
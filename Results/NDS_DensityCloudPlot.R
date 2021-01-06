
# this code is to visualize treatment effect in NDS example.
# treatment effect tau is organized by each covariate levels.

df = read.csv('./tau_nds.csv')


library(ggplot2)
library(ggjoy)
library(ggridges)
library(dplyr)
library(stringr) 


# clean feature name
df$group_old = df$group
df$group = gsub('2_', '_', df$group)
df$group = str_to_title( df$group)
df$group = gsub('_entrance', '', df$group)
df$group = gsub('_junction', '', df$group)
df$group = gsub('Relationtojunction_', 'Junction', df$group)

# order by mean
df2 = df %>% group_by(group) %>% mutate( groupmean = mean(tau)) 
tab = data.frame( unique(df2[ ,c('group', 'groupmean')]) )
tab = tab[order(tab$groupmean),]
df2$group = factor(df2$group, levels = as.character(tab$group ) )


ggplot(data=df2, aes(x = tau, y = group)) + geom_density_ridges() + 
  xlab('Treatment effect') + ylab('') 


ggplot(data=df2[df2$groupmean>0.08,], aes(x = tau, y = group ,fill=groupmean)) + geom_density_ridges() + 
  xlab('Treatment effect') + ylab('') +
  theme(text = element_text(size=25), 
        legend.position="none") 






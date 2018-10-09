A toy-version of cpc implemented in tensorflow based on [the pytorch version given by davidtellez](https://github.com/davidtellez/contrastive-predictive-coding).

Thank @davidtellez for providing network architecture and `data_utils.py`. 

I also directly compute the mutual information between context and future inputs based on "Learning deep representations by mutual information estimation and maximization".

BTW, I do not apply data augmentation, I've implemented one though. Not very sure whether it's about my implementation or something else, it just takes too long to run my local machine :(. Maybe I'll come back to it another day.
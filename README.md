# When To Leave For Work
### Predicting Estimated Departure Time - Boston Subways

##### Problem Statement
Everyday thousands and thousands of commuters travel through the Boston subways. Imagine that you live in Boston, and you have a job near the Harvard Square and you're expected to reach office at `9:00 AM` everyday. You don't want to be late, but you really like to sleep and wake up as late as possible to get the maximum sleep, you can. You stay at an apartment near the stop JFK/U-Mass. It is already known to you that you take `6 minutes` from home to JFK/U-Mass platform and `4 minutes` to walk from Harvard Square stop to work. Given a large historical dataset you need to find out when the trains arrive at JFK/UMass and how long does it take to reach Harvard Square stop. The answers to these vary based on several real life factors, which needs to be analysed.

##### Goal
Our goal is to figure out **What time do we need to leave the house so as to get to work on time, 9 times out of 10 ?** We analyse the past data and patterns from that to build a predictive model that is atleast 90% accurate. Predicting the future is impossible, but we can maximise our chances.

##### Usage
```
git clone https://github.com/amitrajitbose/boston-subway-train-travel-time.git
cd boston-subway-train-travel-time
python3 departure_time.py <YYYY-MM-DD>
```
**Note-1**: *Enter the required date in the format YYYY-MM-DD in the placeholder.*
**Note-2**: *Please satisfy the requirements prior to running the app.*

##### Requirements
```
pandas==0.23.4
numpy==1.15.4
matplotlib==3.0.2
seaborn==0.9.0
requests==2.21.0
```
##### Dataset
The dataset has been collected from the [Massachussets Bay Transport Authority (MBTA)](https://www.mbta.com/) public data repository using the public license and API key. You can request for a private key [here](https://performance.mbta.com/portal).
*[Massachusetts Department of Transportation Developers License Agreement](https://www.mass.gov/files/documents/2017/10/27/develop_license_agree_0.pdf)*

##### Model
We've built a **decision tree classification** algorithm from **scratch** on **Python**, for the problem. We have dealt with multiple real world features like day of the week, season of the year, weekends, etc and other factors that impact the commute time. There can more complex hard-to-visualise features, which can be worked upon in future as an **advancement** to the project.

##### Performance
We divided the dataset we could collect into three partitions. First part was used for training the model, the second part was used to increase its performance by tuning a few hyperparamters like `n-min`, i.e. the minimum number of training examples in a leaf of the decision tree. This is also known as *stopping criterion*. In some problem, `max-depth` is used for the same purpose. Finally, on the third partition we tested our final model.
The following distribution shows how our tested datapoints were mostly on time to get to work or earlier, that is less than or equal to 0.
![performance-distribution](https://raw.githubusercontent.com/amitrajitbose/boston-subway-train-travel-time/master/assets/performance.png)

We used mean error to evaluate the model. We also tried out other methods and here are the best models and reports for them.
| Method | Best n-min | Best Tuning Score | Best Test Score |
|:------:|:----------:|:-----------------:|:---------------:|
| Mean/Averaged Error | 10 | -5.53 | -3.87 |
| Mean Absolute Error | 20 | 6.43 | 5.54 |
| Root Mean Squared Error | 30 | 7.66 | 6.89 |
The mean error had the best parameters, thus we used it in the final version of the model.
##### Acknowledgements
Thanks to Brandon Rohrer for the support.

##### Contribution
Feel free to improve the work and add a PR.

[![](http://alexanderrem.weebly.com/uploads/7/2/5/6/72566533/linkedin_orig.png)](https://www.linkedin.com/in/amitrajitbose/)
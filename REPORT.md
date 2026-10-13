# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Paula Evangelista
- **Student ID**: 20150
- **Email**: pevangelista.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A]

## Dataset
Its a hotel booking demand dataset that I got from kaggle, about two portuguse hotels (city hotel and resort), arrival dates from 2015-07-01 to 2017-08-31 with 32 columns. 

full file is about 6x the 20k row guide, so train.py draws a random sample of 18k rows (seed 42) from both others before splitting. Sampling keep the data coverga of the whole period. after cleaning, 30 zero-guest rows are dropped (train 11,810, val 1,969, test 4,191 rows)

## Business / real-life framing
A revenue manager at a hotel decides how many rooms to overbook. Each overbooking with a high cancellation probability is a bookinh the hotel can overbook against. The ediction moment is booking time. this decision drives wich columns are allowed. Only information known when the booking is made can be used.

Cost (in untis of one night room rate, ADR). flagging a booking means overbooking it.

- Missed cancellation (false negative): the room stays empty, cost = 1.0 night.
- False alarm (false positive): the guest shows up and is walked, cost = 1.5 nights (compensation and goodwill).

These numbers are my own assumptions, not real hotel data. I put them at the top of train.py (COST_FN and COST_FP), and the dashboard reads them from the saved files, so everything uses the same values.


## Data preparation & feature engineering

Removing columns that leak the answer
- reservation_status and reservation_status_date literally say whether the booking was cancelled, so using them would be cheating.
- assigned_room_type is decided close to check-in.
- booking_changes, required_car_parking_spaces and total_of_special_requests can grow after the booking is made.

Cleaning
- children had 4 missing values, which I filled with 0.
- country had 488 missing values, which I filled with "UNK" instead of deleting the rows.
- agent was missing in about 14% of rows. I treated "no agent" as its own category, since it probably means a direct booking.
- - company was missing in about 94% of rows, so I only kept a yes/no column (has_company).
I dropped bookings with zero guests (180 in the full data), because they are not real stays.
- adr (average daily rate) had a negative value and 1,960 rows with a rate of 0 or less. I set negatives to 0 and added a flag column adr_zero.

Splitting the data
I split by arrival date, not randomly: training is July 2015 to December 2016, validation is January to March 2017, and test is April to August 2017. In real life the hotel predicts future bookings from past ones. A random split would mix bookings from the same weeks into train and test, which would make the results look better than they really are.

Feature engineering
- Logs of skewed numbers. Lead time, total nights and previous cancellations are skewed (most values are small, a few are huge). I used log1p so the huge values do not dominate a straight-line model.
- ADR. The maximum ADR was 5,400 while the median was about 95. I capped it at the 99th percentile of the training data, then took the log, and also made ADR per person.
- Simple booking descriptions. Weekend share of nights, number of guests, has children, repeated guest, on the waiting list.
- Month and weekday as sine and cosine. As plain numbers, December (12) and January (1) look far apart, but they are neighbours. Sine and cosine put the months on a circle.
- One-hot encoding for hotel, meal, market segment, distribution channel, room type, deposit type, customer type, country and agent. Country has 177 values and agent has 333, so I only kept the 10 most common of each in the training data and grouped the rest as "OTHER".
- No arrival year. The training data is mostly 2015 to 2016 and the test data is 2017, so the model would have to guess for a year it never saw.
- Scaling. I standardised the continuous columns (mean 0, standard deviation 1), because gradient descent works badly when features have very different scales.

Anything learned from the data (the top-10 lists, the ADR cap, the category list, the scaler) was learned from the training split only and then applied to validation and test. Otherwise information from the test set would leak into the preprocessing. In the end I had 72 features, with identical columns in all three splits.

## Modeling: three implementations, one model

All three are logistic regression, trained on the same features and the same split.

scikit-learn LogisticRegression with L2 regularization. I tried C = 0.01, 0.1, 1 and 10 and picked the best validation logloss (0.4244, 0.4064, 0.4052, 0.4054), so C = 1. The values from 0.1 to 10 are almost the same, so regularization barely matters here.
Manual PyTorch loop. I made the weights w and bias b as tensors with requires_grad=True. Each step computes the scores, applies the sigmoid, computes the cross-entropy loss by hand, calls loss.backward(), and updates the weights inside torch.no_grad(). After each step I reset the gradients, because PyTorch adds new gradients to old ones. I used full-batch gradient descent (all training rows every step) for 20,000 steps.
Standard PyTorch. The same idea with nn.Linear, BCEWithLogitsLoss and torch.optim.SGD, using the same learning rate, steps and regularization.

Making the regularization match. scikit-learn's C is the inverse of the regularization strength, and it adds the penalty to the sum of the losses. My PyTorch versions use the mean loss. To get the same model, the penalty strength is 1 / (C * n), where n is the number of training rows. I did not penalize the bias, because scikit-learn does not either.

Learning rate. I tried 0.3, 0.6 and 1.0. All three gave the same validation logloss (0.4050), so picking 0.3 was just the first of a tie, not a real finding.

Results (threshold 0.5)

All three models are clearly better than the baseline (test logloss 0.686 down to 0.484). The baseline has precision and recall of 0 because it predicts about 37% for everyone, which rounds to "nobody cancels".

Do the three methods agree?

Their scores are the same, but the weights and probabilities were not exactly identical, so I looked into it.

- With 5,000 steps and a smaller learning rate, the biggest weight difference from scikit-learn was 0.84, and the biggest probability difference on the test set was 0.275. With 20,000 steps and larger learning rates, these dropped to 0.137 and 0.050.
- All three methods minimize the same function (the loss plus the penalty), so I compared its value at the end: scikit-learn 0.3730210, manual 0.3730249, nn 0.3730250. scikit-learn's value is slightly lower, so it got a bit closer to the best possible solution. The manual and nn versions match each other almost exactly, so I do not think there is a bug in either one.
- My explanation (this is my best guess, I did not prove it): plain gradient descent stops a little short, because the regularization is very weak and my one-hot columns are correlated (I kept every category instead of dropping one), which makes gradient descent slow. scikit-learn uses a different solver that deals with this better.

Is the deposit feature doing all the work?

Non-refundable bookings were cancelled 99.4% of the time, which looks strange for real life and may be a quirk of the dataset. I retrained scikit-learn without deposit_type. Test AUC went from 0.837 to 0.832 and logloss from 0.484 to 0.498. So the feature helps a bit, but the model does not depend only on it.

Business cost

Cost per booking on the test set, in nights of room revenue:

Policy	Cost
Never flag anyone	0.422
scikit-learn, threshold 0.5	0.295
scikit-learn, threshold 0.65 (the best one on validation)	0.310

At threshold 0.5 on the test set, scikit-learn got 2,181 true negatives, 243 false positives, 873 false negatives and 894 true positives. So the model clearly saves money compared with doing nothing. But the threshold that was best on validation did worse on test than 0.5. I think this is because validation is winter and test is spring and summer, and the cancellation rates are different, but I did not test this directly.

Dashboard
uv run python main.py app opens a Gradio app that only loads the files saved by train.py (the folder models/), so it never retrains anything. It has three tabs: a model comparison (metrics table, predicted probability vs actual cancellation rate for the three models, and each PyTorch model against scikit-learn), the distributions (cancellation rate per split and a histogram of any feature for cancelled vs kept bookings), and a threshold slider that updates the confusion matrix and the business cost


## Limitations & next steps

- Seasonal drift. The cancellation rate is 37.1% in train, 34.0% in validation and 42.2% in test. Performance dropped from validation to test (AUC 0.873 to 0.837). The test set is only one spring and summer, not a full year.
- The threshold did not carry over from validation to test, as shown above.
- The costs are my assumptions. Different real costs would change the best threshold.
- Recall is only about 0.5 at threshold 0.5, so the model misses about half of the real cancellations.
- Limited data. I used 18,000 of 119,390 rows, from two hotels in one country over two years, so the results may not apply to other hotels.
- The deposit feature looks odd, so the scores may look better than they would in the real world.
- Things I did not test. I did not compare the log features with the raw versions, so I cannot claim the logs improved the score. I chose them for the reasons in section 3.
- Linear model. It cannot capture interactions between features.
- The two PyTorch models end up slightly short of the optimum compared to scikit-learn (section 4).

## Generative AI use disclosure

I used Claude (Anthropic's AI assistant) during this assignment. It helped me compare dataset ideas, discuss the framing (predicting at booking time and which columns leak), write the first versions of train.py and app.py, and draft this report. It also suggested the cost values (1.0 and 1.5 nights), which I accepted as assumptions. I picked the hotel dataset myself after rejecting an earlier idea that was not a realistic business problem. I ran all the code on my own computer, checked the outputs, and all the numbers in this report come from my own runs. I read through the code and the report so that I can explain them without the code in front of me.

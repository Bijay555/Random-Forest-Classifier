# Random-Forest-Classifier
This is the beginner understanding knowledge on machine learning famous algo "Random Forest Classifier 

Data science provides a plethora of classification algorithms such as logistic regression, support vector machine, naive Bayes classifier, and decision trees. But near the top of the classifier hierarchy is the random forest classifier

Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction (see figure below).
Visualization of a Random Forest Model Making a Prediction

The fundamental concept behind random forest is a simple but powerful one — the wisdom of crowds. In data science speak, the reason that the random forest model works so well is:

    A large number of relatively uncorrelated models (trees) operating as a committee will outperform any of the individual constituent models.

The low correlation between models is the key. Just like how investments with low correlations (like stocks and bonds) come together to form a portfolio that is greater than the sum of its parts, uncorrelated models can produce ensemble predictions that are more accurate than any of the individual predictions. The reason for this wonderful effect is that the trees protect each other from their individual errors (as long as they don’t constantly all err in the same direction). While some trees may be wrong, many other trees will be right, so as a group the trees are able to move in the correct direction. So the prerequisites for random forest to perform well are:

    There needs to be some actual signal in our features so that models built using those features do better than random guessing.
    The predictions (and therefore the errors) made by the individual trees need to have low correlations with each other.
    
 # Confusion Matrix
 
 A confusion matrix is a table that is often used to describe the performance of a classification model (or “classifier”) on a set of test data for which the true values are known. It allows the visualization of the performance of an algorithm.
    
    A confusion matrix is a summary of prediction results on a classification problem.
The number of correct and incorrect predictions are summarized with count values and broken down by each class. This is the key to the confusion matrix.
The confusion matrix shows the ways in which your classification model is confused when it makes predictions.
It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.
    
    
    data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOsAAACPCAMAAAAssdyrAAAA2FBMVEX///8AAAD09PTx8fGTk5NsbGz8/PwJCQnMzMwQEBCxsbGtra2ZmZni4uK5ubm9vb3ExMTb29vS0tL1+fXD3cM/Pz9OTk77//tlZWVoq2jX6ddZWVno6Oifn5/d693o8+iLi4uAgIAqKio2NjaXvpex0rF3sXe51rnM4cylyaUAgQAdHR2Pu4+Bt4FBm0GIrYhRo1FdpF13uHcsjixmoWY5jTmVxZXR3tFSjVJQmFBBgEFFl0WLw4tAh0AUjBQtfy1ir2Kn0qdxpXGXtZcXfxddj12+0r4AcQDcCPGSAAAJ6klEQVR4nO2cC1ubOhiAv9bJmbd5SWYEt0QdIYkLWEpFuzpd3Y7n//+jE2irVWnaCtat432m84GE5CUXAiQA1NTU1NTU1NTU1NTU1NT8RTT33r8dK4t13fuyVoKvX8vEPtlerOv7tePtDxPZLtz3sPVwf90SOws6eevOUWPRro0Plr0MAUFFWwc4q6tN69GRCfk8PmLDpBfuuj76k3Cfp/TR3rYP/XC4936jc6WHfzVXD8aaHBJaKDkeHfU1SO08OQDpX+Wbdt7QFS6+U95j43tVAGxojx/OgugM/3jsCvia4tgbjy9CQMN4VDxsdfOyflPXlUsE556UHIP0scm7ckMkJCDue17UEYC1ybBUYQd4bvDE1UsQcZXHpXA8beJLPwgA35kDacy6XYFkttXzA9fJzN/UFV0imtCw5yse0muPhygNUesKQu71mCuABzTx7gJkyvWMD1xXn7jSWOrTUPVd5t7hiPEQ9Bn0XRowrUDE5ijSbHUJ/0ne2PVGpCKvd666CHkgjR2EKY0o8VCAnVhfaBGaQu8Acgpd73xFvAhI7EtfxhLuQrg4gxsPKCgfOr6JH5sUXCfrsd7W9btnWqtIHRQLxmhr5JqYCmtcUaQYY63cdcAz1wtGQEYEtUx86F08uELuys3W7zh3hbd2vcz6SqGBhApM4/IhzVxRrAgyRQ2+BsfzQ8I7oPP+tnnw2PU663NkBBC6wFgQZ3XEuLqBuXD5PuiWuYKFLeCuI9Xbujr+pTCVS3cksKD1g9FO6+pW3sbZHyGIXt/ks8XYbRTe0mF7feTqqO++KbC0a/qdMGpTaMXtjjy7pRD32gj3UtDRDwlut93zxM83Lte5eVKuc1K7viIlXY/tY0QrtesrUujq5GN29HzU/gSbq5PFR8iZHPs1XUl2nWBCMJCc3w1Fil1FQIEpQZ7veoTNFfGOZwaHlvP1iq5M/yKAOirQxA20sLmCPHUBuCzY84jmoaUO01MPUm6JXYmr11XJ7eB+hd0ZBlq4/R8CHiGCSPxwtie4pl0JwgPR+oEtCVldgXfZFQLPjTioOHq+vxJXcitQe3ALhXutVjS6G/uGQEVuS5C43eLDhlTsijm+Jthjv4AlrGD/ELsrRK65R4oxO6UxKzhl1dThW+aYUV4Gw1LiUSlmrgkzdxuplL3hCZjg2gdTzWnaBbj0CvYPmeKqOgRoosRPpINXdCVDV5H0eskot5lrG5hLOaDuxWDbRFeWmHDGtWtptVNc8a1ptZGpF4iKgupRkSsduSJzZ8LGytU775uReBym0TDtia6gTimcUhxZetIprsK4Om4gQ5yy5PlhKnG98H1za/z84Mq0UeabrpGkanSaC12R0h4gc3FCacFhHrC7Iu1j0/NzTc3xClr97zGWmJUp5TqF2vUVKel6ZBkDTuPPcl3Zr11no3Z9RWrXGfnDXPdK9cNv7erMQ16uc8UYj5yV64upwPX9u3nYPG4czBXhMSeHL4+7uVqmSuVsNhr/zM7a2jyhn8WeK62nNBobpV13nebMOO8bG3MEf8Kn/aOVF0d2Nipw3ZoneMm+6V2Jvml90a4lrzl/k+vmyyP/aa4ffxdXcv9QDR491JbXYw+6lsQVX48eizE8Lsvi5XNth9k8JMk5SWMhFaO+gzBXgFzsqNEz4+VwRVcsYOC5suepgMrEY7+ciw7tSeJiFI2KdjlchasTAakG6oiQQM+D74ggGnHjCveXipKun0vktELXHwBcZ7PszP8hQbmrdLmrMtd7lsLV65ufHhMtjyLhUhIJdgOhghaHCtvrb+FKVGD6pVg7fpAi6qaEB7hzJzX2texo5I666GVwzSeuIvOLDN6Rmx8yeNGN8hfeowa7FK4zUtK1TG5r13moXSeyNK5Mn5thsastwa2uiNILBA6lEybETHHNJw1RNukWt+Jy5TchgCbDZIuwuvZv/r1koJLzCWfL6urIawUQdnv9CQGqduXnEjRyBPcnzGexuqbZBBMUSfKr+FStfLHklgaRD14CctLskqpdMY+RJtJlXlScot2164aIJRRuiidNWF0RMuNT3BvMmCiialeB3DuNUjP274jC4PY6HKIzhc4Z3BRHtrrmK17uWsB6i3IFL+4RZVzj4kpsdaUIdBslHtwUZ3e6q0xMNV6Ma5pm81kIbUvRekEdVl3RuYMw1HFxRzzF1VwAUFfpYEK/WK0rEgoB4+YmQHBaHNzqSpQvCDDlTyiZKa5CZqtx1KSJb1WPJZwn/z+j3Fji5PcZS8xA7Tojteus/E2u24t/T2db6zuF5myujtNcWVn5tGXY3d3d28uXyO/sHFXgupcd2Eae6F6e5t5G42h7e7goe30j5/OQh1fgR4bjo2PD6uqB4XD/cH9//+PHk8bXk5NszXpjXgYvqsu6vmt8MZxkeZgnE9krdhPDxDvJ4n807OccZhyMsfrAwdr+8T3mfORTAzaHp2pjjPUBo+Xu29s7Ozvv31XgOkr13eYozUFC609X2GdpbjaOdrKEh+wNMFVtdyvHVANTS1buefRqfMvU4bHJHhlzvI/9UEV7fZy6nVLtdeXr39MPf3pd1629KUdYHlfn88GUmrk8rtCcNhF7iVynslDXtdp1NhbhStDYU4YlcUW4+OGH1x5bNrYkrvJ+cdlL58GIb99uJjy7ybG74uxBk3vuE5C9XsFD1ypdA53NH2BCkrQrKEYMO8QTOJ8Hg63rrkakyp6W1TW9PmcQBsgVcI3l9fNLZZXzYELqMqAx7sm+y0Tief85MvASb/Z5MKmPpe1y/ukfS25l9rD0UoD2aYLQ+fPllhW6ci260mQXJBrOg/mGkJStfG7I/XpCu2vsd2xrfbesuUW/GPwrwfelcS2oxBW6tinVIQSP5sHQKxzz2efBUAZ+25LWrjW3zLheYlOupoBJ7zXLlaamx41Rtt4WeEAgMq5O6EOgsnkwfLb22ocz2wvN6a5BAAGHU+mdvmJ7ddJsHsxlCLoVUdZqOypOL6UMRZDSjo+SmdorS5JJD+1zprheMiBxohHgJCloChXOgxl9IuL+B9Dwn/l1fw2yX18d+ycm7K75V+UGn6Yr+EDdnzZG3C21EqN2nYe/1TV/5mRvc0vjynw/mythC740rkSdmqGibeSzPK5AIpx9xQ0HAUN9t6iAl8cV5GlKs2+66EB2yJK7wrXOhvB33Gedws9iLJNrbKqwGS46gLwgLAi+TK4t48pOMVXKw62C4KVcS70gqdz1ohOamylTpEwGhd8DWyLXqdSuM1K7zsrCXUuta357193mysw0dxobcwR/EtlUihdHXmlW4Xq4OgeHjf15glcYeXW/tOv2vNn9WCK7pSjvOucXFLbWd1/81YSSbK3P1bPU1NTU1NTU1NTU1NTU1NTULJz/AQxvWjEtTf98AAAAAElFTkSuQmCC
    
# Datasets:
the datasets is most common machine learning dataset "Iris Datasets"

# Assign url of file: url
iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

Originally published at UCI Machine Learning Repository: Iris Data Set, this small dataset from 1936 is often used for testing out machine learning algorithms and visualizations (for example, Scatter Plot). Each row of the table represents an iris flower, including its species and dimensions of its botanical parts, sepal and petal, in centimeters.
    
    

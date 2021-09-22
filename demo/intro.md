### Business Problem

Suppose Amazon wants to add a million new products from hundreds of different suppliers. Each product has data provided by its supplier; it has a title, brand, description, and category. Amazon needs to assimilate those products into its existing classification scheme. This could take days of tedious human labor, or minutes of machine time.

### Demonstration

To address this problem, I've developed a text classifier which infers a product's Amazon category with 97% accuracy. Let's try it on products from another website. The following data was scraped from [Walmart](https://www.walmart.com/) in 2019, and is available on [Kaggle](https://www.kaggle.com/promptcloud/walmart-product-data-2019).

Use the slider to select a Walmart category, then click **Random Product** to iterate through products in that category. Feel free to modify the product data or enter your own. When you're satisfied, click **Amazon Classify**.
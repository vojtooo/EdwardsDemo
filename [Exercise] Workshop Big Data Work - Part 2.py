# Databricks notebook source
# MAGIC %md
# MAGIC <div style="width: 100%; background: black; height: 96px">
# MAGIC   <img src="https://s3-us-west-2.amazonaws.com/curriculum-release/images/db_learning_rev.png" style="display: block; margin: auto"/>
# MAGIC </div>
# MAGIC # [EXERCISE] Spark Workshop
# MAGIC ## PART 2 - (py)spark SQL module

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) 1 Import

# COMMAND ----------

# Import only what you need
#from pyspark.sql.functions import col, lit, collect_list #...

# Or (imho better), whole modules
import pyspark.sql.functions as f
from pyspark.sql.window import Window
import pyspark.sql.types as T

# COMMAND ----------

# MAGIC %md ##![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) 2 Read data

# COMMAND ----------

# Gapminder dataset
df_gapminder = (
    spark
    .table('demo.gapminder_basic_stats')
)

# Nobel dataset
df_nobel = (
    spark
    .table('demo.nobel')
)

# COMMAND ----------

display(df_gapminder)

# COMMAND ----------

display(df_nobel)

# COMMAND ----------

# MAGIC %md ##![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) 3 Exercises

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 [EXERCISE] PySpark SQL/DataFrame functions
# MAGIC Difficulty 3/10 - 10min

# COMMAND ----------

# DBTITLE 1,Quest: various operations
# QUEST: Take the Gapminder dataset and go through the following tasks:
# 1. Get the "summary" of the dataframe
# 2. Get maximum and minimum of the `pop` column for `continent == "Asia"`
# 3. Apply `log` function on the `year` column
# 4. Create a new column `adv_col` and let it equal to either 1 (if lifeExp > 50) or 0 (if lifeExp <= 50)
# 5. BONUS: calculate correlation between `year` and `pop`

# COMMAND ----------

# -----------------
#      ANSWER
# -----------------

# 1. Get the "summary" of the dataframe
df_gapminder.summary()

# 2. Count maximum and minimum of the `pop` column
df_gapminder.filter(f.col('continent') == 'Asia').agg(f.max('pop'), f.min('pop'))

# 3. Apply `log` function on the `year` column
df_gapminder.select(f.log('year'))

# 4. Create a new column `adv_col` which equals to either 1 (if lifeExp > 50) or 0 (if lifeExp <= 50)
df_gapminder.withColumn('adv_col', f.when(f.col('lifeExp') > 50, 1).otherwise(0)) # good solution
df_gapminder.withColumn('adv_col', (f.col('lifeExp') > 50).cast('int')) # but this is more concise :)

# 5. BONUS: calculate correlation between `year` and `pop`
df_gapminder.corr('pop', 'year')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 [EXERCISE] Aggregations
# MAGIC Difficulty 8/10 - 15 min, then help

# COMMAND ----------

display(df_nobel)

# COMMAND ----------

# DBTITLE 1,Quest: find the number of Nobel prizes per category
# Nobel dataset
# QUEST: Find the total number of Nobel prizes for each category.

# COMMAND ----------

df_laureate_number = (df_nobel
    .withColumn('nlaureates', f.size('laureates'))
    .groupBy('category')
    .count()
     )

display(df_laureate_number)

# COMMAND ----------

# DBTITLE 1,Quest: prize sharing per category
# Nobel dataset
# QUEST: Find which category has the largest average number of laurates sharing the prize

# Hint: there is a pyspark.sql.function for array length

# COMMAND ----------

# -----------------
#      ANSWER
# -----------------

df_laur_stat = (
    df_nobel
    .withColumn('nlaureates', f.size('laureates'))
    .groupBy('category')
    .agg(
        f.avg('nlaureates').alias('nlaureates_avg')
    )
    .orderBy(f.desc('nlaureates_avg'))
)

display(df_laur_stat)

# COMMAND ----------

# DBTITLE 1,Quest: life exp increase
# Gapminder Dataset: Recall that in the Gapminder dataset, there are measurements of `lifeExp`ectancy. They are measured in 5-year periods (1952, 1957, 1962, ...), for each country. 
# QUEST: Find which country had the highest increase in life expectancy between any two consecutive "5-years". 


# Hint: 
# - pyspark.sql.window.Window, f.lag

# COMMAND ----------

window = Window().partitionBy('country').orderBy('year')

df_gapminder_lagged = (df_gapminder
               .withColumn('lifeExp_lag', f.lag('lifeExp').over(window))
               .withColumn('lifeExp_difference', f.col('lifeExp') - f.col('lifeExp_lag'))
               .orderBy('lifeExp_difference', ascending=False) 
               )

display(df_gapminder_lagged)

# COMMAND ----------

# DBTITLE 1,Quest: Bulgaria's population growth
# There is population count every 5 years in the Gapmidner dataset. We want to however find the average of population growth curve taking into account the last 20 years (4 rows before). Use average over window with rowsBetween.

# COMMAND ----------

window = Window().partitionBy('country').orderBy('year')

display(df_gapminder
           .withColumn('population_mov_avg', f.avg('pop').over(window.rowsBetween(-4, 0)))
           .orderBy('year', ascending=True)
            .select('country', 'year', 'pop', 'population_mov_avg')
            .filter(f.col('country') == 'Bulgaria')
       )

# COMMAND ----------

# DBTITLE 1,Bonus Quest: Bulgaria - sabotaged edition
# There has been a sabotage and the years might contain null values, meant to change the shape of the population growth curve. Rewrite the previous approach, so that Null values are not skipped.

# COMMAND ----------

window = Window().partitionBy('country').orderBy('year')

display(df_gapminder
           .withColumn('population_mov_avg', f.avg('pop').over(window.rangeBetween(Window.currentRow - 20, 0)))
           .orderBy('year', ascending=True)
            .select('country', 'year', 'pop', 'population_mov_avg')
            .filter(f.col('country') == 'Bulgaria')
       )

# COMMAND ----------

# DBTITLE 1,Quest: Get the average value
# Print out the average of column lifeExp as a number. Use collect and then round this number to 1 decimal place.

# COMMAND ----------

print(round(df_gapminder.select('lifeExp').groupBy().avg().collect()[0][0], 2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 [EXERCISE] UDFs
# MAGIC Difficulty 4/10 - 10 min

# COMMAND ----------

# DBTITLE 1,Quest: UDF replacing letters
# QUEST: Create UDF that replaces all 'a' letters with 'L' letter. Apply it to the Gapminder dataset (to the column `country`).  E.g. "Afghanistan" will become "AfghLnistLn"
# Hint: Python native function: string.replace()

# Bonus question: could this be done via SparkSQL function? 

# COMMAND ----------

@udf(T.StringType())
def replace_a(string):
    return string.replace('a', 'L')

display(
    df_gapminder
    .select('country', replace_a(f.col('country')).alias('country_renamed'))
)

# COMMAND ----------

display(
    df_gapminder
    .withColumn('country_renamed', f.regexp_replace(f.col('country'), '[aA]', 'L'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 [EXERCISE] Complex type manipulation 
# MAGIC (Difficulty: 9/10) - 15min, then help

# COMMAND ----------

# DBTITLE 1,Quest: array, struct manipulation
# QUEST: Take the Nobel dataframe and create an identical dataframe with just one difference - in the `laureates` column, keep ONLY the "firstname" attribute in the structs and discard the rest (=> no "id", "motivation", "share" or "surname" in the structs of the `laureates` array). 
# That is, the resulting display should look like
# +--------+-----------------+----+-------------------------------------------------------------------------------------------+
# |category|overAllMotivation|year|laureates                                                                                  |
# +--------+-----------------+----+-------------------------------------------------------------------------------------------+
# |medicine|null             |1950|[{"firstname": "Edward Calvin}, {"firstname": "Tadeus"}, {"firstname": "Philip Showalter"}]|
# |peace   |null             |1908|[{"firstname": "Klas Pontus}, {"firstname": "Fredrik"}]                                    |
# +--------+-----------------+----+-------------------------------------------------------------------------------------------+

# Hints: 
# - split+tranform+combine approach; with `struct`, `collect_list` functions
# - (or use udf)

# COMMAND ----------

# DBTITLE 1,BOOKS DATASET
# BOOKS-RATINGS
df_books_ratings = (
    spark
    .table('demo.bx_ratings')
)

# BOOKS-BOOKS
df_books_books = (
    spark
    .table('demo.bx_books')
)

# BOOKS-USERS
df_books_users = (
    spark
    .table('demo.bx_users')
)

# COMMAND ----------

# DBTITLE 1,Book ratings - fact table
display(df_books_ratings)

# COMMAND ----------

print(df_books_ratings.count())

# COMMAND ----------

# DBTITLE 1,Book info - dim table
display(df_books_books)

# COMMAND ----------

# DBTITLE 1,User info - dim table
display(df_books_users)

# COMMAND ----------

# DBTITLE 1,Quest: Join (15 mins)
# Please fill each subtask into individual cell(s).

## SUBTASKS
# 1. Pair the ratings with the book info and user info

# 2. Find the name of the book with the highest average rating

# 3. Find the name of the book with the lowest average rating

# 4. Create histogram of all the ratings

# 5. Create histogram of the ages of the individual users

# COMMAND ----------

# 1. Pair the ratings with the book info and user info

# COMMAND ----------

# 2. Find the name of the book with the highest average rating

# COMMAND ----------

# 3. Find the name of the book with the lowest average rating

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Quest: Join2
# QUEST: compute pairwise differences in population among countries in 2007

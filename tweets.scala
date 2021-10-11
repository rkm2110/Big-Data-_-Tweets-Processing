import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.feature.{HashingTF, Tokenizer, StringIndexer, StopWordsRemover}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{Row, SparkSession, SaveMode}
import scala.runtime.ScalaRunTime._




object tweets {
  def main(args: Array[String]): Unit = {

    if (args.length != 2) {
      println("Usage: WordCount InputDir OutputDir")
    }
    val config = new SparkConf().setAppName("Tweet Sentiment classification")

    // create Spark context with Spark configuration
    val spark = new SparkSession.Builder().getOrCreate()
    import spark.implicits._

    //Read data from csv and convert to DataFrames
    val tweets = spark.read.option("inferSchema","true").option("header","true").csv(args(0)).toDF()

    //Step 1 : Select needed columns and drop null values
    val tweets1 = tweets.select("tweet_id", "text", "airline_sentiment").na.drop()

    //Split data into training and test
    val Array(training, test) = tweets1.randomSplit(Array(0.9, 0.1))

    // Configure an ML pipeline

    //Step 2a : Tokenizing raw text to words
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    ////Step 2b : Remove stop-words
    val stop_words_removed = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol(("text_without_stopwords"))

    //Step 2c : Work Token to vector (feature hashing)
    val hashingTF = new HashingTF()
      .setInputCol(stop_words_removed.getOutputCol)
      .setOutputCol("features")

    //Step 2d : Convert the string to index
    val indexer = new StringIndexer()
      .setInputCol("airline_sentiment")
      .setOutputCol("label") // Index "airline_sentiment" as "label"


    // Step 3 : Creating a model of Logistic Regression (Training)
    val lr = new LogisticRegression()
      .setMaxIter(10)

    // Create a pipeline of steps 2a, 2b, 2c, 2d and lr
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stop_words_removed, hashingTF, indexer, lr))


    //Step 4 : We use a ParamGridBuilder to construct a grid of parameters to search over.
    // With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
    // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()


    // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    // This will allow us to jointly choose parameters for all Pipeline stages.
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator) // We have more than 2 labels so can't use binaryClassificationEvaluator
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)  // Use 3+ in practice.
    //.setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

    // Run cross-validation, and choose the best set of parameters.
    // Will to 6 iterations
    val cvModel = cv.fit(training)

    //Check predictions on test data
    val result = cvModel.bestModel.transform(test)
    //display(result.select("text", "label", "prediction"))

    //Convert to RDD format for metrics evaluation
    val result_evaluation = result
      .select("label", "prediction")
      .rdd
      .map{ case Row(label: Double, prediction: Double) => (prediction, label)}


    //Step 5 : Evaluation metrics

    // Instantiate metrics object
    val metrics = new MulticlassMetrics(result_evaluation)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)




    // Overall Statistics
    val accuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")



    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }

    // F-measure by label metrics.fMeasure(labels(0)) labels[0,1,2]
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }

    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")


    val output = Seq(
      ("Precision",metrics.weightedPrecision.toString),
      ("Accuracy",metrics.accuracy.toString),
      ("Recall",metrics.weightedRecall.toString),
      ("FMeasure",metrics.weightedFMeasure.toString),
      ("True Positive Rate",metrics.weightedTruePositiveRate.toString),
      ("False Positive Rate",metrics.weightedFalsePositiveRate.toString)

    ).toDF("Evaluation Metrics", "Value")

    output.coalesce(1).write.option("header", "true").format("csv").mode(SaveMode.Overwrite).save(args(1))

  }

}

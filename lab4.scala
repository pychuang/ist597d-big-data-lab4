import java.io.File
import java.io.PrintWriter
import java.io.Serializable

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext

import scala.collection.mutable
import scala.collection.immutable.ListMap
import scala.collection.immutable.Seq

class Lab4 extends Serializable {
  def work() = {
    val configuration = new Configuration()
    configuration.addResource(new Path("/usr/hdp/2.3.0.0-2557/hadoop/conf/core-site.xml"))
    configuration.addResource(new Path("/usr/hdp/2.3.0.0-2557/hadoop/conf/core-site.xml"))

    val lines = sc.textFile(FileSystem.get(configuration).getUri + "/ist597j/Iris/Iris-comma-sep-2.csv")
    val parsedData = lines.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

    // Cluster the data
    val numClusters = 3
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    val result = parsedData.map(x => "(" + x.toArray.mkString(",") + ")," + clusters.predict(x)).collect()

    // output
    outputResult(result)
    outputCenters(clusters.clusterCenters)
  }

  def outputResult(data: Array[String]): Unit = {
    val outputFile = "output.txt"
    val writer = new PrintWriter(new File(outputFile))
    data.foreach(writer.println)
    writer.close()
  }

  def outputCenters(centers: Array[Vector]): Unit = {
    val outputFile = "centers.txt"
    val writer = new PrintWriter(new File(outputFile))
    centers.map(v => v.toArray.mkString(",")).foreach(writer.println)
    writer.close()
  }
}

new Lab4().work()

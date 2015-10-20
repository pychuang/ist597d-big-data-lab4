import java.io.File
import java.io.PrintWriter
import java.io.Serializable

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

import scala.collection.mutable

import breeze.linalg.DenseVector

class Lab4 extends Serializable {
  def work() = {
    val configuration = new Configuration()
    configuration.addResource(new Path("/usr/hdp/2.3.0.0-2557/hadoop/conf/core-site.xml"))
    configuration.addResource(new Path("/usr/hdp/2.3.0.0-2557/hadoop/conf/core-site.xml"))

    val lines = sc.textFile(FileSystem.get(configuration).getUri + "/ist597j/Iris/Iris-comma-sep-2.csv")
    val points = lines.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

    // Cluster the data
    val numClusters = 3
    val numIterations = 20

    var centers = randomlyPickCenters(points, numClusters)
    for (i <- 1 to numIterations) {
      centers = calcNewCenters(centers, points)
    }

    // output
    val result = points.map(p => "(" + p.toArray.mkString(",") + ")," +  nearestCenterIndex(centers, p)).collect()

    outputResult(result)
    outputCenters(centers.toArray)
  }

  def randomlyPickCenters(points: RDD[Vector], numClusters: Int): Array[Vector] = {
    return points.takeSample(false, numClusters)
  }

  def nearestCenterIndex(centers: Array[Vector], point: Vector): Int = {
    return centers.map(c => Vectors.sqdist(point, c)).zipWithIndex.min._2.toInt
  }

  def genClusters(currentCenters: Array[Vector], points: RDD[Vector]): RDD[Array[Vector]] = {
    return points.map(p => (nearestCenterIndex(currentCenters, p), p)).groupByKey().map(_._2.toArray)
  }

  def calcNewCenter(points: Array[Vector]): Vector = {
    val c = points.map(p => new DenseVector(p.toArray)).reduce(_+_) / points.toList.length.toDouble
    return Vectors.dense(c.toArray)
  }

  def calcNewCenters(currentCenters: Array[Vector], points: RDD[Vector]): Array[Vector] = {
    val clusters = genClusters(currentCenters, points)
    return clusters.map(calcNewCenter).toArray
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

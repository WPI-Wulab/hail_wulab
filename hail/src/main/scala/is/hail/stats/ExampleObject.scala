// This file contains the example object from PHowell's instruction file hail-wulab-install-instructions.model.
// It is an example showing how to run a Scala script file inside Hail by java, after compiling.
// Created by ZWu, 2025-01-27.


package is.hail.stats
import is.hail.methods.gfisher.StatGFisher //used in the example
import breeze.linalg._ // used in the example

object ExampleObject {

  def main(args: Array[String]): Unit = {
    // insert code you want to run here
    val p = DenseVector(0.04, 0.01)
    val df = DenseVector(2, 2)
    val w = DenseVector(0.7, 0.3)
    val result = StatGFisher.statGFisher(p, df, w)
    println(result)
  }

}

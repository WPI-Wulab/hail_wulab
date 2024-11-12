package is.hail.utils

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}


object ArrayToMatrix {

  /**
    * Convert an array of vectors to a matrix. Column major order.
    *
    * @param a array containing
    */
  def arrayToMatrix(a: Array[BDV[Double]]): BDM[Double] = {
    val m = a.length
    val n = a(0).size
    val AData = new Array[Double](m*n)
    var i = 0
    while (i < m) {
      System.arraycopy(a(i).data, 0, AData, i*n, n)
      i += 1
    }
    val A = new BDM[Double](n,m, AData)
    return A
  }



}

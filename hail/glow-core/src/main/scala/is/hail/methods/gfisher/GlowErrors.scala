/**
  * Error and warning helpers for the GLOW numerical core, so it needs no Hail code. They replace
  * is.hail.utils.fatal and is.hail.utils.warn in the core files; the adapters (GFisher.scala,
  * GLOW.scala, PrepData.scala) keep Hail's. Core files import them explicitly
  * (import is.hail.methods.gfisher.GlowErrors.fatal), so the names cannot clash with the wildcard
  * import of is.hail.utils._ in the adapters.
  *
  * Behavior differs from Hail's in two ways, both accepted in the plan: fatal throws
  * IllegalArgumentException rather than HailException, so Python reports a Java error rather than
  * Hail's FatalError; warn writes to stderr rather than Hail's log. No GLOW code catches
  * HailException. These are the definitions GLOWh's packages/glow-core harness shim used, so the
  * fixture suite has run against them.
  * Last update (latest update first):
  *   Claude Code (Opus 5.5), prompted by VPanwar, 2026-11-03: created, stage 1 of GLOWh's
  *     0project/glow-hail-refactor-plan.md.
  */

package is.hail.methods.gfisher

object GlowErrors {

  /** Abort with a message: an input or a result the method cannot use. */
  def fatal(msg: String): Nothing = throw new IllegalArgumentException(msg)

  /** Report a condition the method has corrected automatically (e.g. equal weights substituted). */
  def warn(msg: String): Unit = System.err.println(s"WARN: $msg")
}

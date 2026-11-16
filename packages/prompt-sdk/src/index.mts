/** Private Node-only portable consumer; no compiler, loader, credentials or model transport. */
export {
  canonicalBytes, consumeBuild, consumeRender, hashDomain, parsePortableJSON, renderBuild,
  type ConsumedBuild, type ConsumedRender,
} from "./kernel.mts";
export {
  MAX_CONSUMED_BUILD_BYTES, consumePinnedBuild, renderConsumed,
  type ConsumeOptions, type LocalBuildReference, type LocalBuildSource, type PinnedBuild,
} from "./consumption.mts";
export {
  MAX_ATTEMPTS, MAX_BODY_BYTES, MAX_ATTEMPT_METADATA_BYTES, MAX_INVOCATION_METADATA_BYTES,
  PromptCaptureError, LocalByteResponse, PreparedInvocation, CapturedExchange,
  prepareInvocation, captureOnce, validateAttemptMetadata, validateInvocationMetadata,
  type CaptureCode, type AttemptGap, type AttemptParents, type DeclaredBodyPin,
  type ByteObservation, type AttemptMetadata, type InvocationMetadata, type LocalByteTransport,
} from "./attempts.mts";

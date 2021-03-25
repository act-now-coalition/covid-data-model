// Track page view conversion event.
// Default gtag plugin doesn't support conversion events - this
// script is included in head to track conversion events.
gtag("event", "conversion", { send_to: "AW-527465415/lRyqCM2a9-kBEMf3wfsB" });

// Track API Signup
function gtag_report_conversion() {
  gtag("event", "conversion", {
    send_to: "AW-527465415/TkWKCNqspv0BEMf3wfsB",
  });
  return false;
}

// Track page view conversion event.
// Default gtag plugin doesn't support conversion events - this
// script is included in head to track conversion events.
gtag("event", "conversion", { send_to: "AW-527465415/lRyqCM2a9-kBEMf3wfsB" });

function gtag_report_conversion(url) {
  var callback = function () {
    if (typeof url != "undefined") {
      window.location = url;
    }
  };
  gtag("event", "conversion", {
    send_to: "AW-527465415/TkWKCNqspv0BEMf3wfsB",
    event_callback: callback,
  });
  return false;
}

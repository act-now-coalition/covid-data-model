---
id: about
title: Covid Act Now API
sidebar_label: About
slug: /
---

Welcome to the documentation for the Covid Act Now API.

These docs describe version 2 of the API.
Documentation for the previous version of the API is available [here](https://github.com/covid-projections/covid-data-model/blob/master/api/README.V1.md).

The API provides the same data that powers [Covid Act Now](https://covidactnow.org)
but in an easily digestible, machine readable format, intended for consumption by other COVID websites, models, and tools.

:::note

If you are interested in using the Covid Act Now API, please
[register for an API Key](/access).

:::

### Highlights

- 5 key COVID metrics that have been standardized for consistency across states and counties for easy comparison, specifically: Daily New Cases per 100K, Infection Growth Rate (Rt), Test Positivity Rate, ICU Headroom, and Tracers Hired.

  We believe this set of metrics provides a more complete and reliable picture of COVID especially given the uncertainty and misreporting that’s been reporting. The metrics also help contextualize each other. For instance, if you have a very high Test Positivity rate but a low number of Daily New Cases per 100K, then chances are you’re not testing enough, and/or the incidence is higher than you think.

- A comprehensive compilation of the latest U.S. state and county COVID data, including cases, hospitalizations, hospital capacity, hospital utilization (by both COVID and non-COVID patients), deaths, testing, and tracer numbers.

  This data is aggregated from the best-available official, public and private sources; it is secured and vetted by our dedicated team and advisory board of public health, epidemiology, public policy experts. When necessary, the data is scraped directly from official state and county COVID dashboards, and verified against other sources for veracity.

- COVID data that is regularly verified, sanitized, and updated, generally on a daily-basis, by our team of engineers, modelers, and analysts.

- An R<sub>t</sub> (infection growth rate) that is optimized for capturing COVID dynamics at the county-level. <!-- Read more about how it’s calculated here. -->

- Our simple yet robust green-yellow-orange-red [COVID risk-level framework](https://globalepidemics.org/wp-content/uploads/2020/09/key_metrics_and_indicators_v5-1.pdf), researched and developed in conjunction with the Harvard Global Health Institute (HGHI) and Harvard Edmund J. Safra Center for Ethics.

### License

Data is licensed under [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/).

We are a volunteer-driven 501(c)3 non-profit whose mission is to provide reliable and trustworthy COVID data,
with a focus on U.S. counties. Since we started on March 20th, we’ve made our model and data open-source
and freely available to the public.

This data requires a non-trivial amount of work to organize, sanitize, maintain, and develop,
especially given the fast-changing nature of COVID and health data in the U.S. In order to
ensure that we can continue to deliver best-in-class open-source COVID data, our tiny,
volunteer-driven team appreciates donations here.

We also ask that, as legally required by our license, all commercial entities wishing to
access our API contact <info@covidactnow.org> to acquire a commercial license. We define
commercial users as any individual or entity engaged in commercial activities, such as selling goods or services.

Our non-commercial users can freely download, use, share, modify, or build upon our source code.

import React from "react";
import clsx from "clsx";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import useBaseUrl from "@docusaurus/useBaseUrl";
import styles from "./styles.module.css";
import { RedocStandalone } from "redoc";
import open_api_schema from '../../open_api_schema.json';



function APIReference() {
  const context = useDocusaurusContext();
  const { siteConfig = {} } = context;
  return (
    <Layout title={siteConfig.title} description="API Reference Documentation">
      <RedocStandalone 
          spec={open_api_schema}
          options={{
            disableSearch: false,
            hideDownloadButton: true,
            hideHostname: false,
            hideSingleRequestSampleTab: true,
            expandSingleSchemaField: false,
            /* expandResponses: "all", */
            pathInMiddlePanel: true,
            scrollYOffset: 60,
            menuToggle: true
          }}
        />
    </Layout>
  );
}

export default APIReference;

module.exports = {
  title: "Covid Act Now",
  tagline: "API Documentation",
  url: "https://apidocs.covidactnow.org",
  baseUrl: "/",
  plugins: ['@docusaurus/plugin-google-analytics'],
  onBrokenLinks: "throw",
  favicon: "img/favicon.ico",
  organizationName: "covid-projections", // Usually your GitHub org/user name.
  projectName: "covid-data-model", // Usually your repo name.
  themeConfig: {
    googleAnalytics: {
      trackingID: 'UA-160622988-1',
    },
    navbar: {
      title: "",
      logo: {
        alt: "Covid Act Now Logo",
        src: "img/can_logo.png",
      },
      items: [
        {
          to: "/",
          activeBaseRegex: "/(?!(api))",
          label: "Guide",
          position: "left",
        },
        {
          to: "api",
          activeBasePath: "api",
          label: "API Reference",
          position: "left",
        },

        {
          href: "https://github.com/covid-projections/covid-data-model",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Docs",
          items: [
            {
              label: "API Reference",
              to: "/",
            },
          ],
        },
        {
          title: "Community",
          items: [
            {
              label: "Facebook",
              href: "https://www.facebook.com/covidactnow",
            },
            {
              label: "Instagram",
              href: "https://www.instagram.com/covidactnow",
            },
            {
              label: "Twitter",
              href: "https://twitter.com/CovidActNow",
            },
          ],
        },
        {
          title: "More",
          items: [
            {
              label: "Blog",
              href: "https://blog.covidactnow.org/",
            },
            {
              label: "Covid Act Now",
              href: "https://covidactnow.org",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Covid Act Now. Built with Docusaurus.`,
    },
  },
  presets: [
    [
      "@docusaurus/preset-classic",
      {
        docs: {
          sidebarPath: require.resolve("./sidebars.js"),
          routeBasePath: "/",
        },
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      },
    ],
  ],
};

import logging
import os
import enum
from typing import Optional

import sentry_sdk
import structlog

# from structlog_sentry import SentryJsonProcessor


# env variable holding the Sentry Environment name
SENTRY_ENVIRONMENT_ENV = "SENTRY_ENVIRONMENT"


class Environment(str, enum.Enum):
    """Identifies an environment, as used at Sentry.io."""

    PRODUCTION = "production"
    DEVELOPMENT = "development"
    STAGING = "staging"  # Not used as of 2020-06-23


def configure_logging(command: Optional[str] = None):
    """Configure stdlib logging and structlog, and if SENTRY_DSN is set, Sentry.

    Parameters:
        command: a command name added to the Sentry events.
    """

    # First structlog is configured to send errors to Sentry and use stdlib for console logging. If we start
    # getting duplicate logs from structlog I'm guessing stdlib is also logging them so remove SentryProcessor.

    # Based on https://www.structlog.org/en/stable/standard-library.html#rendering-using-structlog-based-formatters-within-logging
    structlog.configure(
        context_class=dict,
        wrapper_class=structlog.stdlib.BoundLogger,
        processors=[
            structlog.stdlib.add_log_level,  # required before SentryProcessor()
            # sentry_sdk creates events for level >= ERROR. Getting breadcrumbs from structlog isn't supported
            # without a lot of custom work. See https://github.com/kiwicom/structlog-sentry/issues/25.
            # The SentryJsonProcessor is used to protect against event duplication.
            # It adds loggers to a sentry_sdk ignore list, making sure that the message logged is
            # not reported in addition to the exception stack trace.
            # SentryJsonProcessor(level=logging.ERROR, tag_keys="__all__"),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # It is important that wrap_for_formatter is the last processor. It converts the processed
            # event dict to something that the ProcessorFormatter (the logging.Formatter passed to
            # setFormatter) understands.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Second, configure stdlib logging to format structlog events and any other parameters we want to set.
    formatter = structlog.stdlib.ProcessorFormatter(
        # TODO(tom) fix https://github.com/hynek/structlog/issues/166 so keys appear in added order
        processor=structlog.dev.ConsoleRenderer(),
        foreign_pre_chain=[
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # Initialize sentry_sdk.
    sentry_dsn = os.getenv("SENTRY_DSN")
    sentry_environment = None
    if SENTRY_ENVIRONMENT_ENV in os.environ:
        sentry_environment = Environment(os.getenv(SENTRY_ENVIRONMENT_ENV))

    if sentry_dsn:
        sentry_sdk.init(sentry_dsn, environment=sentry_environment)

        if command:
            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("command", command)
                # changes applied to scope remain after scope exits. See
                # https://github.com/getsentry/sentry-python/issues/184

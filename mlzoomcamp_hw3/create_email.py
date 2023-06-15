from time import sleep
from prefect import flow
from prefect.context import get_run_context
from prefect_email import EmailServerCredentials, email_send_message

def notify_exc_by_email(exc):
    context = get_run_context()
    flow_run_name = context.flow_run.name
    email_server_credentials = EmailServerCredentials.load("email-server-credentials")
    email_send_message(
        email_server_credentials=email_server_credentials,
        subject=f"Flow run {flow_run_name!r} failed",
        msg=f"Flow run {flow_run_name!r} failed due to {exc}.",
        email_to=email_server_credentials.username,
    )

@flow
def example_flow():
    try:
        1 / 0
    except Exception as exc:
        notify_exc_by_email(exc)
        raise


if __name__ == "__main__":
    example_flow()
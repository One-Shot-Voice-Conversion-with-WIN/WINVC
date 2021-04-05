from tensorboardX import SummaryWriter


# Logger使用
class Logger(object):
    """Using tensorboardX such that need no dependency on tensorflow."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

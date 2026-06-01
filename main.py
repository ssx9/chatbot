from agent import deps, weather_agent
from grdio_ui import GradioUI

if __name__ == '__main__':
    ui = GradioUI(agent=weather_agent, deps=deps)
    ui.demo.launch(width='100%', css=ui.FULLSCREEN_CSS, auth=('admin', '123456'))

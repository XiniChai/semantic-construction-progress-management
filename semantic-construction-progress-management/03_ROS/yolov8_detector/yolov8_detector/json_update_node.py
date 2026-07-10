# json_update_node.py
import rclpy, json, os, time
from rclpy.node import Node
from std_msgs.msg import String
from datetime import datetime

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def duration_str(sec):
    if sec is None:
        return None
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}m{s}s" if m>0 else f"{s}s"

class JsonUpdater(Node):
    def __init__(self):
        super().__init__('json_update_node')
        self.declare_parameter('state_file', os.path.expanduser('~/components_state.json'))
        self.declare_parameter('complete_state', ['installed', 'done', 'completed'])
        self.state_file = self.get_parameter('state_file').get_parameter_value().string_value
        self.complete_state = self.get_parameter('complete_state').get_parameter_value().string_array_value

        self.sub = self.create_subscription(String, 'yolo/step_filtered', self.cb, 10)
        self.pub = self.create_publisher(String, 'state/json', 10)
        # load existing
        self.state = {}
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file,'r') as f:
                    self.state = json.load(f)
            except:
                self.state = {}
        self.get_logger().info("JsonUpdater ready")

    def cb(self, msg):
        data = json.loads(msg.data)
        for comp in data.get("components", []):
            cid = comp.get("component_id")
            state = comp.get("state")
            conf = comp.get("conf")
            if cid is None:
                continue
            entry = self.state.get(cid, {})
            # if first time seen -> set start
            if 'hasActualStart' not in entry:
                entry['hasActualStart'] = now_iso()
            # update status and conf
            prev_state = entry.get('hasState')
            entry['hasState'] = state
            entry['conf'] = conf
            entry['bbox'] = comp.get('bbox')
            entry['last_seen'] = now_iso()
            # if now status is a completion status and previous wasn't, set end and duration
            if state in self.complete_state and prev_state not in self.complete_state:
                entry['hasActualEnd'] = now_iso()
                # compute duration
                try:
                    st = datetime.fromisoformat(entry['hasActualStart'].replace('Z',''))
                    ed = datetime.fromisoformat(entry['hasActualEnd'].replace('Z',''))
                    dur = (ed - st).total_seconds()
                    entry['hasDuration'] = duration_str(dur)
                    entry['hasDurationSeconds'] = int(dur)
                except Exception:
                    entry['hasDuration'] = None
            self.state[cid] = entry
        # save file and publish
        with open(self.state_file,'w') as f:
            json.dump(self.state, f, indent=2)
        self.pub.publish(String(data=json.dumps({"ts": time.time(), "state": self.state})))
        self.get_logger().info(f"State updated: {len(self.state)} components")

def main(args=None):
    rclpy.init(args=args)
    node = JsonUpdater()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


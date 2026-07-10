import rclpy, threading, json, time
from rclpy.node import Node
from std_msgs.msg import String
from flask import Flask, jsonify, render_template_string
import logging

APP = Flask(__name__)

class VizNode(Node):
    def __init__(self):
        super().__init__('viz_node')
        self.state = {}
        self.ttl = ""
        self.last_update = time.strftime("%Y-%m-%d %H:%M:%S")

        # ROS订阅
        self.sub_state = self.create_subscription(String, 'state/json', self.cb_state, 10)
        self.sub_ttl = self.create_subscription(String, 'rdf/ttl', self.cb_ttl, 10)

        # Flask运行线程
        t = threading.Thread(target=self.run_flask, daemon=True)
        t.start()
        self.get_logger().info("✅ Flask UI running at http://0.0.0.0:5000")

    # ===== 回调函数 =====
    def cb_state(self, msg):
        try:
            payload = json.loads(msg.data)
            self.state = payload.get("state", payload)
            self.last_update = time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            self.get_logger().warn(f"State parse error: {e}")

    def cb_ttl(self, msg):
        self.ttl = msg.data

    # ===== Flask路由 =====
    def run_flask(self):
        
        # ✅ 关闭 Flask HTTP request log（关键）
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        log.disabled = True   # ← 推荐加这一行（更彻底）

        @APP.route('/')

        def index():
            return render_template_string("""
            <html>
            <head>
                <title>Semantic Construction Dashboard</title>
                <style>
                    body { font-family: Arial; background-color: #f4f4f4; margin: 30px; }
                    pre { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 0 5px #ccc; }
                    h1, h2 { color: #333; }
                    #status { 
                        position: fixed; 
                        top: 10px; 
                        right: 20px; 
                        background: #4CAF50; 
                        color: white; 
                        padding: 6px 12px; 
                        border-radius: 6px; 
                        display: none;
                    }
                </style>
            </head>
            <body>
                <h1>🏗️ Construction Semantic Dashboard</h1>
                <p>Last Update: <b id="update_time">{{update_time}}</b></p>

                <h2>📦 Component States (JSON)</h2>
                <pre id="json_data">{{state}}</pre>

                <h2>🔗 RDF Graph (TTL Format)</h2>
                <pre id="ttl_data">{{ttl}}</pre>

                <div id="status">✅ Updated</div>

                <script>
                    async function fetchData() {
                        const [stateResp, ttlResp] = await Promise.all([
                            fetch('/data/state'),
                            fetch('/data/ttl')
                        ]);
                        const stateData = await stateResp.json();
                        const ttlData = await ttlResp.json();

                        document.getElementById('json_data').textContent = JSON.stringify(stateData.state, null, 2);
                        document.getElementById('ttl_data').textContent = ttlData.ttl;
                        document.getElementById('update_time').textContent = stateData.time;

                        let status = document.getElementById('status');
                        status.style.display = 'block';
                        setTimeout(() => status.style.display = 'none', 1000);
                    }

                    // 每5秒刷新一次
                    setInterval(fetchData, 5000);
                </script>
            </body>
            </html>
            """, 
            state=json.dumps(self.state, indent=2, ensure_ascii=False),
            ttl=self.ttl,
            update_time=self.last_update)

        # JSON数据接口
        @APP.route('/data/state')
        def data_state():
            return jsonify({
                "state": self.state,
                "time": self.last_update
            })

        # TTL数据接口
        @APP.route('/data/ttl')
        def data_ttl():
            return jsonify({
                "ttl": self.ttl
            })

        APP.run(host='0.0.0.0', port=5000, debug=False)


def main(args=None):
    rclpy.init(args=args)
    node = VizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import XSD
import os, json


class RDFNode(Node):
    def __init__(self):
        super().__init__("rdf_node")

        # =========================
        # config
        # =========================
        self.declare_parameter("semantic_base_dir", "/home/xinichai/semantic_base")
        self.semantic_base_dir = self.get_parameter("semantic_base_dir").value
        os.makedirs(self.semantic_base_dir, exist_ok=True)

        self.base_graph_path = os.path.join(self.semantic_base_dir, "building_graph.ttl")
        self.updated_graph_path = os.path.join(self.semantic_base_dir, "building_graph_updated.ttl")

        # =========================
        # RDF graph
        # =========================
        self.graph = Graph()

        if os.path.exists(self.base_graph_path):
            try:
                self.graph.parse(self.base_graph_path, format="turtle")
                self.get_logger().info("Base TTL loaded.")
            except Exception as e:
                self.get_logger().error(f"TTL parse error: {e}")

        self.CPMB = Namespace("http://www.semanticweb.org/cmpb/")
        self.DICE = Namespace("https://w3id.org/digitalconstruction/Entities#")
        self.DICP = Namespace("https://w3id.org/digitalconstruction/0.5/Processes#")
        self.DICA = Namespace("https://w3id.org/digitalconstruction/0.5/Agents#")
        self.BOT = Namespace("https://w3id.org/bot#")
        self.BRICK = Namespace("https://brickschema.org/schema/Brick#")
        self.CSITE = Namespace("http://www.owl-ontologies.com/cSite#")
        self.graph.bind("cpmb", self.CPMB)
        self.graph.bind("dice", self.DICE)
        self.graph.bind("dicp", self.DICP)
        self.graph.bind("dica", self.DICA)
        self.graph.bind("bot", self.BOT)
        self.graph.bind("brick", self.BRICK)
        self.graph.bind("cSite", self.CSITE)

        # =========================
        # ROS IO
        # =========================
        self.sub = self.create_subscription(String, "state/json", self.cb_state, 10)
        self.pub_ttl = self.create_publisher(String, "rdf/ttl", 10)

        self.get_logger().info("rdf_node ready.")

    # =========================================================
    # callback
    # =========================================================
    def cb_state(self, msg: String):

        try:
            payload = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"JSON parse error: {e}")
            return

        if not isinstance(payload, dict):
            return

        state_dict = payload.get("state", payload)

        if not state_dict:
            return

        for cid, info in state_dict.items():

            # =========================================
            # 1. component URI (SQM_xx)
            # =========================================
            component_uri = self.CPMB[cid]

            process_id = info.get("process_id", None)
            process_uri=None

            if process_id:
            # =========================================
            # 2. normalize process ID → P1_03
            # =========================================
                if "|" in process_id:
                    pid = process_id.split("|")[0]
                elif "__" in process_id:
                    pid = process_id.split("__")[0]
                else:
                    pid = process_id

                process_uri = self.CPMB[pid]
            

            # =========================================
            # 3. link component → process
            # =========================================
            self.graph.set((
                component_uri,
                self.DICE.hasActivity,
                process_uri
            ))

            # =========================================
            # 4. process-level time (IMPORTANT)
            # =========================================

            if "hasActualStart" in info and info["hasActualStart"]:
                self.graph.set((
                    process_uri,
                    self.CSITE.hasActualStart,
                    Literal(info["hasActualStart"], datatype=XSD.dateTime)
                ))

            if "hasActualFinish" in info and info["hasActualFinish"]:
                self.graph.set((
                    process_uri,
                    self.CSITE.hasActualFinish,
                    Literal(info["hasActualFinish"], datatype=XSD.dateTime)
                ))

            if "hasActualDuration" in info and info["hasActualDuration"]:
                self.graph.set((
                    process_uri,
                    self.CSITE.hasActualDuration,
                    Literal(info["hasActualDuration"], datatype=XSD.string)
                ))

            # =========================================
            # 5. component state (only component level)
            # =========================================
            if "hasState" in info and info["hasState"]:
                state_uri = self.CPMB[str(info["hasState"])]
                self.graph.set(
                    (
                        component_uri,
                        self.DICE.hasState,
                        state_uri
                    ))

            # =========================================
            # 6. confidence (component level)
            # =========================================
            if "conf" in info:
                try:
                    self.graph.set((
                        component_uri,
                        self.CPMB.confidence,
                        Literal(float(info["conf"]))
                    ))
                except Exception:
                    pass

        # =========================================
        # save TTL
        # =========================================
        ttl=self.graph.serialize(format="turtle")
        try:
            os.makedirs(os.path.dirname(self.updated_graph_path), exist_ok=True)
            self.graph.serialize(self.updated_graph_path, format="turtle")
        except Exception as e:
            self.get_logger().error(f"TTL save error: {e}")

        # publish TTL
        try:
            self.pub_ttl.publish(String(data=ttl))
        except Exception as e:
            self.get_logger().error(f"TTL publish error: {e}")


# =========================
# main
# =========================
def main(args=None):
    rclpy.init(args=args)
    node = RDFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()






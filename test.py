from otx.backend.native.engine import OTXEngine

if __name__ == "__main__":
    engine = OTXEngine(
        model="src/otx/recipe/detection/atss_mobilenetv2.yaml",
        data="tests/assets/car_tree_bug",
    )
    engine.train(max_epochs=10)

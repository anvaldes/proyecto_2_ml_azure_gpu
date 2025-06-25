from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, command, dsl
from azure.ai.ml.entities import Environment, BuildContext, AmlCompute
from azure.ai.ml.constants import AssetTypes

# 1. Conexión al workspace
print("🔌 Conectando al workspace de Azure ML...")
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="88e891ec-0f1c-4cba-9ed0-2dafc3c69861",
    resource_group_name="rg-proyecto1",
    workspace_name="ws-proyecto1-v2"
)
print("✅ 1. Conectado al Workspace")

# 2. Crear cluster CPU si no existe
cluster_name = "gpu-cluster"


print(f"⏳ Creando cluster '{cluster_name}'")
ml_client.compute.begin_create_or_update(
    AmlCompute(
        name=cluster_name,
        size="Standard_NC4as_T4_v3",
        min_instances=0,
        max_instances=2,
        idle_time_before_scale_down=120
    )
).result()
print(f"✅ 2. Cluster '{cluster_name}' creado exitosamente")

# 3. Crear environment desde Dockerfile
print("🐳 Creando environment desde Dockerfile local...")
env = Environment(
    name="hf-distributed-env",
    build=BuildContext(path="."),
)
env = ml_client.environments.create_or_update(env)
print(f"✅ 3. Environment '{env.name}' creado con versión: {env.version}")

# 4. Definir componente de entrenamiento
print("🧩 Definiendo componente de entrenamiento...")
train_component = command(
    name="train_transformers_distributed",
    code=".",
    command="python train.py --data_dir ${{inputs.data_dir}} --output_dir ${{outputs.output_dir}}",
    inputs={
        "data_dir": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml://datastores/pipelineblobstore_2/paths/datasets/2025_06"
        )
    },
    outputs={
        "output_dir": Output(type=AssetTypes.URI_FOLDER)
    },
    environment=env.id,
    compute=cluster_name
)
print("✅ 4. Componente de entrenamiento definido")

# 5. Crear pipeline DSL
print("🧪 Creando pipeline DSL...")
@dsl.pipeline(
    compute=cluster_name,
    description="HF Distributed Training Pipeline with Evaluation Logs"
)
def hf_pipeline():
    train_step = train_component()
    return {"output_dir": train_step.outputs.output_dir}

print("✅ 5. Pipeline DSL creado")

# 6. Ejecutar pipeline
print("🚀 Lanzando pipeline...")
pipeline_job = ml_client.jobs.create_or_update(hf_pipeline())
print(f"✅ 6. Pipeline lanzado con ID: {pipeline_job.name}")
print(f"📊 Monitorea la ejecución en:\nhttps://ml.azure.com/experiments/{pipeline_job.display_name or 'pipeline'}/runs/{pipeline_job.name}")

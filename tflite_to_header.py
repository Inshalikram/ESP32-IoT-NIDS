print("🚀 Generating tinyml_model.h...")

# =========================================================
# READ TFLITE
# =========================================================

with open("models/tinyml_final.tflite", "rb") as f:
    data = f.read()

# =========================================================
# WRITE HEADER
# =========================================================

with open("models/tinyml_model.h", "w") as f:

    f.write("#ifndef TINYML_MODEL_H\n")
    f.write("#define TINYML_MODEL_H\n\n")

    f.write(f"// Auto-generated from tinyml_final.tflite\n")
    f.write(f"// Size: {len(data)} bytes ({len(data)/1024:.2f} KB)\n\n")

    f.write(f"const unsigned int tinyml_model_len = {len(data)};\n\n")

    # alignas(8) required — TFLite Micro needs 8-byte alignment on ESP32
    # without this the model will crash at runtime
    f.write("alignas(8) const unsigned char tinyml_model[] = {\n  ")

    for i, byte in enumerate(data):

        f.write(f"0x{byte:02x},")

        if (i + 1) % 12 == 0:
            f.write("\n  ")

    f.write("\n};\n\n")

    f.write("#endif // TINYML_MODEL_H\n")

print("✅ tinyml_model.h generated")
print(f"📦 Size : {len(data)/1024:.2f} KB")
print(f"📄 File : models/tinyml_model.h")
print(f"\nIn your Arduino sketch add:")
print(f'  #include "tinyml_model.h"')
print(f"  const tflite::Model* model = tflite::GetModel(tinyml_model);")
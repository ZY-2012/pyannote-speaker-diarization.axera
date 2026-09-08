// AX650 axengine wrapper (named multi-IO, cached buffers + cache maintenance).
// Model is loaded into host memory; AX_ENGINE_Init called once with VNPU disabled.
#include "EngineWrapper.hpp"
#include "ax_sys_api.h"

#include <cstdlib>
#include <fstream>
#include <vector>

#if !defined(AX630C)
static int ax_engine_init_once() {
    static bool inited = false;
    if (inited) return 0;
    AX_ENGINE_NPU_ATTR_T attr;
    memset(&attr, 0, sizeof(attr));
    attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    int ret = AX_ENGINE_Init(&attr);
    if (ret == 0) inited = true;
    return ret;
}
#endif

EngineWrapper::EngineWrapper()
    : m_hasInit(false), m_handle(nullptr), m_io_info(nullptr),
      m_input_num(0), m_output_num(0) {
    memset(&m_io, 0, sizeof(m_io));
}

EngineWrapper::~EngineWrapper() { Release(); }

int EngineWrapper::Init(const char* strModelPath, uint32_t nNpuType, const char* axclConfig) {
    (void)nNpuType;
    (void)axclConfig;
    std::ifstream file(strModelPath, std::ios::binary);
    if (!file.is_open()) {
        printf("Read model(%s) fail\n", strModelPath);
        return -1;
    }
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0);
    std::vector<char> model((size_t)size);
    file.read(model.data(), size);
    file.close();

    AX_ENGINE_NPU_ATTR_T attr;
    memset(&attr, 0, sizeof(attr));
    attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    if (ax_engine_init_once() != 0) {
        printf("AX_ENGINE_Init fail\n");
        return -1;
    }

    if (AX_ENGINE_CreateHandle(&m_handle, model.data(), (AX_U32)model.size()) != 0 ||
        !m_handle) {
        printf("Create model(%s) handle fail\n", strModelPath);
        return -1;
    }
    if (AX_ENGINE_CreateContext(m_handle) != 0) {
        printf("CreateContext fail\n");
        return -1;
    }
    if (AX_ENGINE_GetIOInfo(m_handle, &m_io_info) != 0) {
        printf("GetIOInfo fail\n");
        return -1;
    }
    m_input_num = m_io_info->nInputSize;
    m_output_num = m_io_info->nOutputSize;
    m_input_name_to_idx.clear();
    m_output_name_to_idx.clear();
    for (int i = 0; i < m_input_num; ++i)
        m_input_name_to_idx[std::string(m_io_info->pInputs[i].pName)] = i;
    for (int i = 0; i < m_output_num; ++i)
        m_output_name_to_idx[std::string(m_io_info->pOutputs[i].pName)] = i;

    m_io.nInputSize = m_input_num;
    m_io.nOutputSize = m_output_num;
    m_io.pInputs = new AX_ENGINE_IO_BUFFER_T[m_input_num]();
    m_io.pOutputs = new AX_ENGINE_IO_BUFFER_T[m_output_num]();
    for (int i = 0; i < m_input_num; ++i) {
        AX_U32 bytes = m_io_info->pInputs[i].nSize;
        m_io.pInputs[i].nSize = bytes;
        if (AX_SYS_MemAllocCached(&m_io.pInputs[i].phyAddr, &m_io.pInputs[i].pVirAddr,
                                  bytes, 128, (const AX_S8*)"in") != 0) {
            printf("input alloc fail\n");
            return -1;
        }
        memset(m_io.pInputs[i].pVirAddr, 0, bytes);
    }
    for (int i = 0; i < m_output_num; ++i) {
        AX_U32 bytes = m_io_info->pOutputs[i].nSize;
        m_io.pOutputs[i].nSize = bytes;
        if (AX_SYS_MemAllocCached(&m_io.pOutputs[i].phyAddr, &m_io.pOutputs[i].pVirAddr,
                                  bytes, 128, (const AX_S8*)"out") != 0) {
            printf("output alloc fail\n");
            return -1;
        }
    }
    m_hasInit = true;
    return 0;
}

int EngineWrapper::SetInput(void* pInput, int index) {
    if (!m_hasInit || index < 0 || index >= m_input_num) return -1;
    memcpy(m_io.pInputs[index].pVirAddr, pInput, m_io.pInputs[index].nSize);
    return 0;
}

int EngineWrapper::RunSync() {
    if (!m_hasInit) return -1;
    for (int i = 0; i < m_input_num; ++i)
        AX_SYS_MflushCache(m_io.pInputs[i].phyAddr, m_io.pInputs[i].pVirAddr,
                           m_io.pInputs[i].nSize);
    int ret = AX_ENGINE_RunSync(m_handle, &m_io);
    if (ret != 0) printf("AX_ENGINE_RunSync failed. ret=0x%x\n", ret);
    for (int i = 0; i < m_output_num; ++i)
        AX_SYS_MinvalidateCache(m_io.pOutputs[i].phyAddr, m_io.pOutputs[i].pVirAddr,
                                m_io.pOutputs[i].nSize);
    return ret;
}

int EngineWrapper::GetOutput(void* pOutput, int index) {
    if (!m_hasInit || index < 0 || index >= m_output_num) return -1;
    memcpy(pOutput, m_io.pOutputs[index].pVirAddr, m_io.pOutputs[index].nSize);
    return 0;
}

int EngineWrapper::SetInputByName(const char* name, void* pInput) {
    auto it = m_input_name_to_idx.find(std::string(name));
    if (it == m_input_name_to_idx.end()) {
        printf("Input '%s' not found in model\n", name);
        return -1;
    }
    return SetInput(pInput, it->second);
}

int EngineWrapper::GetOutputByName(const char* name, void* pOutput) {
    auto it = m_output_name_to_idx.find(std::string(name));
    if (it == m_output_name_to_idx.end()) {
        printf("Output '%s' not found in model\n", name);
        return -1;
    }
    return GetOutput(pOutput, it->second);
}

int EngineWrapper::GetInputSize(int index) {
    if (index < 0 || index >= m_input_num) return -1;
    return m_io.pInputs[index].nSize;
}
int EngineWrapper::GetOutputSize(int index) {
    if (index < 0 || index >= m_output_num) return -1;
    return m_io.pOutputs[index].nSize;
}
int EngineWrapper::GetInputSizeByName(const char* name) { return GetInputSize(GetInputIndex(name)); }
int EngineWrapper::GetOutputSizeByName(const char* name) { return GetOutputSize(GetOutputIndex(name)); }

int EngineWrapper::GetInputIndex(const char* name) {
    auto it = m_input_name_to_idx.find(std::string(name));
    return it == m_input_name_to_idx.end() ? -1 : it->second;
}
int EngineWrapper::GetOutputIndex(const char* name) {
    auto it = m_output_name_to_idx.find(std::string(name));
    return it == m_output_name_to_idx.end() ? -1 : it->second;
}
const char* EngineWrapper::GetInputName(int index) const {
    return (index >= 0 && index < m_input_num) ? m_io_info->pInputs[index].pName : "";
}
const char* EngineWrapper::GetOutputName(int index) const {
    return (index >= 0 && index < m_output_num) ? m_io_info->pOutputs[index].pName : "";
}
const char* EngineWrapper::GetInputDtypeStr(int index) const { return "float32"; }
const char* EngineWrapper::GetOutputDtypeStr(int index) const { return "float32"; }

int EngineWrapper::Release() {
    if (!m_hasInit) return 0;
    for (int i = 0; i < m_input_num; ++i)
        if (m_io.pInputs[i].pVirAddr)
            AX_SYS_MemFree(m_io.pInputs[i].phyAddr, m_io.pInputs[i].pVirAddr);
    for (int i = 0; i < m_output_num; ++i)
        if (m_io.pOutputs[i].pVirAddr)
            AX_SYS_MemFree(m_io.pOutputs[i].phyAddr, m_io.pOutputs[i].pVirAddr);
    delete[] m_io.pInputs;
    delete[] m_io.pOutputs;
    memset(&m_io, 0, sizeof(m_io));
    if (m_handle) AX_ENGINE_DestroyHandle(m_handle);
    m_handle = nullptr;
    m_hasInit = false;
    return 0;
}

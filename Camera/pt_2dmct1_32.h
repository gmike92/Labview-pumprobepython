#include "extcode.h"
#pragma pack(push)
#pragma pack(1)

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
	int32_t CenterRow;
	int32_t ROIHeight;
} Cluster;

/*!
 * PT_2DMCT_Close
 */
int32_t __cdecl PT_2DMCT_Close(void);
/*!
 * PT_2DMCT_GetConfig
 */
int32_t __cdecl PT_2DMCT_GetConfig(int32_t Config[], int32_t len);
/*!
 * PT_2DMCT_GetFPATemp
 */
int32_t __cdecl PT_2DMCT_GetFPATemp(char TempKStr[], double *TempK, 
	int32_t len);
/*!
 * PT_2DMCT_GetFrames
 */
int32_t __cdecl PT_2DMCT_GetFrames(int32_t NumFrames, int32_t NumExtChs, 
	int16_t AcqTrigger, int32_t Timeout, uint16_t FrameData[], 
	double ExtChanData[], int32_t *RowsOut, int32_t lenFrameData, 
	int32_t lenExtChanData);
/*!
 * PT_2DMCT_Initialize
 */
int32_t __cdecl PT_2DMCT_Initialize(int16_t FrameTrigger);
/*!
 * PT_2DMCT_SetFrameSize
 */
int32_t __cdecl PT_2DMCT_SetFrameSize(uint16_t FrameSize, 
	double *MaxFrameRate);
/*!
 * PT_2DMCT_SetIntegration
 */
int32_t __cdecl PT_2DMCT_SetIntegration(double IntUs, double *ActualUs);
/*!
 * PT_2DMCT_SetOffset
 */
int32_t __cdecl PT_2DMCT_SetOffset(int32_t GainLevelDepreciated, 
	int32_t OffsetLevel);
/*!
 * PT_SetROI
 */
int32_t __cdecl PT_SetROI(LVBoolean UseROIs, Cluster ROIs[], int32_t len);

MgErr __cdecl LVDLLStatus(char *errStr, int errStrLen, void *module);

void __cdecl SetExecuteVIsInPrivateExecutionSystem(Bool32 value);

#ifdef __cplusplus
} // extern "C"
#endif

#pragma pack(pop)


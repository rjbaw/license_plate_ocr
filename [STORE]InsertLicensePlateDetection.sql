USE [CornSilo]
GO

/****** Object:  StoredProcedure [dbo].[InsertLicensePlateDetection]    Script Date: 02/01/2020 19:43:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE PROCEDURE [dbo].[InsertLicensePlateDetection]
	@Location			nvarchar(10),
	@LicensePlateNumber	nvarchar(10),
	@NO1Confident		float,
	@NO2Confident		float,
	@NO3Confident		float,
	@NO4Confident		float,
	@NO5Confident		float,
	@NO6Confident		float,
	@LicensePlateText	nvarchar(5),
	@TextConfident		float,
	@TruckType			nvarchar(10),
	@Direction			tinyint
AS
BEGIN

	INSERT INTO LicensePlateDetection
           ([RecordedTime]
           ,[Location]
           ,[LicensePlateNumber]
           ,[NO1Confident]
           ,[NO2Confident]
           ,[NO3Confident]
           ,[NO4Confident]
           ,[NO5Confident]
           ,[NO6Confident]
           ,[LicensePlateText]
           ,[TextConfident]
           ,[TruckType]
           ,[Direction])
     VALUES
           (GETDATE()
           ,@Location
           ,@LicensePlateNumber
           ,@NO1Confident
           ,@NO2Confident
           ,@NO3Confident
           ,@NO4Confident
           ,@NO5Confident
           ,@NO6Confident
           ,@LicensePlateText
           ,@TextConfident
           ,@TruckType
           ,@Direction)

END

GO


